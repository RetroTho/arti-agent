"""browser: Arti's own headless Chrome, driven over the DevTools protocol (see devtools.py).

The tool: one tool with a small set of actions. navigate and the other page actions answer with a
snapshot: the page text, with each link, button and field tagged like `[e12] button "Sign in"`,
which click and type then name. Each chat context and profile has its own Chrome (at most
MAX_BROWSERS at once), closed after 120 s unused and at shutdown. The default profile is a fresh,
throw-away one; a named profile is one the owner signed in to by hand (`python -m arti
browser-login NAME`, at the end of this file), and acting on it, navigating included, needs
approval.

Keeping the local network out of reach (Chrome.handle_event does most of it):
- Every HTTP request Chrome makes, from any tab, popup, frame or worker, is paused until it has
  been checked with the safety plugin's address rules (net.py). Each new tab, frame or worker also
  starts paused, and is set up (Chrome._set_up) before it may run. The first tab, opened before any
  of this was switched on, is replaced.
- Chrome looks the name up again itself, so the address each response actually came from is
  checked too: a local one (DNS rebinding) resets the browser, though that one request has been
  sent by then.
- WebSockets, which the request check cannot see, are switched off in every tab, frame and
  worker; shared workers, which that switch does not reach, are never let start. Chrome runs with
  no proxy (so a response's address is the server's own), WebRTC without UDP, no downloads.
Also: page text is capped and marked untrusted, page dialogs are dismissed, and Arti never types
into password, card or one-time-code fields.
"""

import argparse
import base64
import itertools
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from arti import fsutil
from arti.tools import ToolError, UnknownOutcome
from plugins.safety import net, proc, redact

from .devtools import COMMAND_SECONDS, POLL, BrowserError, Cancelled, DevTools, WebSocket

REQUIRES = ("safety",)  # imports from it, and its gate reviews this plugin's tools

CHROMES = ("chromium", "google-chrome", "google-chrome-stable", "chromium-browser")  # on PATH
CHROME_FLAGS = (
    "--remote-debugging-address=127.0.0.1 --remote-debugging-port=0 --no-first-run "
    "--no-default-browser-check --disable-background-networking --disable-component-update "
    "--disable-sync --disable-extensions --disable-default-apps --disable-breakpad "
    "--disable-crash-reporter --password-store=basic --use-mock-keychain --window-size=1280,900 "
    "--no-proxy-server "  # connect directly, so a response's address is the server's own
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"  # the request check misses WebRTC
).split()
# Password saving, autofill and sign-in stay off in every profile Arti starts.
PREFERENCES = {
    "credentials_enable_service": False,
    "profile": {"password_manager_enabled": False},
    "signin": {"allowed": False},
    "autofill": {"profile_enabled": False, "credit_card_enabled": False},
    "browser": {"check_default_browser": False},
}
IDLE_SECONDS = 120
MAX_BROWSERS = 3
LOAD_SECONDS = 15
MAX_EVALUATE_CHARS = 100_000
PROFILE_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
SCOPE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # a place's id (its contexts/ folder's name)
ACTIONS = "navigate snapshot click type press scroll back screenshot evaluate close".split()
CHANGES_PAGE = ("navigate", "click", "type", "press", "back")  # wait for the page to load after
# Actions that may change something on a site: reviewed on a signed-in profile (navigate too),
# and an unknown outcome when they are interrupted.
ACTS_ON_SITE = ("click", "type", "press", "evaluate")
# Targets that make requests: each one is set up (Chrome._set_up) before it may run.
TARGET_TYPES = ("page", "iframe", "worker", "service_worker")
# The request check does not see WebSockets, so they are cut off: "offline" for ws and wss.
NO_WEBSOCKETS = [
    {
        "urlPattern": f"{scheme}://*:*/*",
        "offline": True,
        "latency": 0,
        "downloadThroughput": -1,
        "uploadThroughput": -1,
    }
    for scheme in ("ws", "wss")
]
# The keys press knows, with the Windows virtual key codes CDP wants.
KEYS = {
    "Enter": 13,
    "Tab": 9,
    "Escape": 27,
    "Backspace": 8,
    "ArrowUp": 38,
    "ArrowDown": 40,
    "ArrowLeft": 37,
    "ArrowRight": 39,
    "PageUp": 33,
    "PageDown": 34,
}
SCROLLS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}
SETTINGS = {"executable": ""}  # [browser]: Chrome's binary; "" = found (chrome_path)


# --- starting Chrome ------------------------------------------------------------------------


def chrome_path(app) -> str:
    """browser.executable, else the first Chromium or Chrome on PATH ("" when there is none: the
    browser tools stay hidden)."""
    if configured := app.config["browser"]["executable"]:
        return configured
    return next(filter(None, map(shutil.which, CHROMES)), "")


def prepare_profile(profile: Path) -> None:
    prefs_path = fsutil.private_dir(profile / "Default") / "Preferences"
    try:
        prefs = json.loads(prefs_path.read_text())
    except (OSError, ValueError):
        prefs = {}
    for key, value in PREFERENCES.items():
        if isinstance(value, dict):
            prefs.setdefault(key, {}).update(value)
        else:
            prefs[key] = value
    fsutil.write_json(prefs_path, prefs)
    (profile / "DevToolsActivePort").unlink(missing_ok=True)  # else we read the last run's


def start_chrome(executable: str, profile: Path, scratch: Path, headless: bool = True):
    """Start Chrome on profile; returns (proc.Child, the browser's DevTools URL)."""
    prepare_profile(profile)
    home = fsutil.private_dir(scratch / "home")
    env = redact.bare_env(HOME=str(home), TMPDIR=str(fsutil.private_dir(scratch / "tmp")))
    headless_flag = ["--headless=new"] if headless else []
    argv = [executable, *headless_flag, f"--user-data-dir={profile}", *CHROME_FLAGS, "about:blank"]
    child = proc.spawn(argv, cwd=scratch, env=env, stdout=proc.DEVNULL, stderr=proc.DEVNULL)
    # Chrome writes the port it picked, and the browser's path, into the profile.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and child.proc.poll() is None:
        try:
            port, path = (profile / "DevToolsActivePort").read_text().split("\n")[:2]
            if port.isdigit() and path.startswith("/devtools/browser/"):
                return child, f"ws://127.0.0.1:{port}{path}"
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    child.kill()
    raise BrowserError("Chrome did not start (is another Chrome using this profile?)")


# --- one Chrome, and its request checks -----------------------------------------------------


class Chrome(DevTools):
    """One Chrome process with one page, driven over one DevTools connection."""

    def __init__(self, executable: str, profile: Path, scratch: Path, allow_private: bool):
        self.scratch, self.allow_private = scratch, allow_private
        self.lock = threading.Lock()  # one tool call at a time
        self.last_used = time.monotonic()
        self.notes: list[str] = []  # dialogs dismissed and requests blocked, for the next answer
        self.pages: dict[str, str] = {}  # target id -> session, of every page set up
        self._guards: set[int] = set()  # ids of the WebSocket cut-offs, which must not fail
        self.child, endpoint = start_chrome(executable, profile, scratch)
        try:
            super().__init__(endpoint)
            # Enabled on the browser, not the page, so it also pauses the requests of popups,
            # cross-site frames and workers (a page-level Fetch misses popups).
            self.cmd("Fetch.enable", patterns=[{"urlPattern": "*"}])
            self.cmd("Browser.setDownloadBehavior", behavior="deny")
            # Every target from now on starts paused until handle_event has set it up.
            self.cmd(
                "Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=True, flatten=True
            )
            # The first page opened before Fetch.enable, which misses its requests: replace it.
            targets = self.cmd("Target.getTargets")["targetInfos"]
            page = self.cmd("Target.createTarget", url="about:blank")["targetId"]
            for target in targets:
                if target["type"] == "page":
                    self.cmd("Target.closeTarget", targetId=target["targetId"])
            deadline = time.monotonic() + COMMAND_SECONDS
            while page not in self.pages:
                if time.monotonic() > deadline:
                    raise BrowserError("Chrome did not attach its page")
                self.pump(POLL)
            self.session = self.pages[page]  # from now on, cmd() talks to this page
        except BaseException:
            self.close()
            raise

    def handle_event(self, message: dict) -> None:
        method, params = message.get("method"), message.get("params", {})
        session = message.get("sessionId")  # answer on the session the event came from
        if message.get("id") in self._guards:
            self._guards.discard(message["id"])
            if "error" in message:
                raise BrowserError("could not cut off WebSockets in a new tab, frame or worker")
        if method == "Fetch.requestPaused":
            url, request = params["request"]["url"], {"requestId": params["requestId"]}
            if self._allowed(url):
                self.send("Fetch.continueRequest", request, session)
                return
            if url.startswith(("http:", "https:")):
                self.notes.append(f"blocked a request to {url[:200]} (local/private address)")
            self.send("Fetch.failRequest", {**request, "errorReason": "AccessDenied"}, session)
        elif method == "Target.attachedToTarget":
            self._set_up(params["sessionId"], params["targetInfo"])
        elif method == "Network.responseReceived":
            address = params["response"].get("remoteIPAddress", "").strip("[]")
            if address and not self._address_ok(address):  # no address: from the cache
                url = params["response"].get("url", "")[:200]
                raise BrowserError(
                    f"{url} was answered from the local/private address {address} although "
                    "its name was vetted as public (DNS rebinding?)"
                )
        elif method == "Page.javascriptDialogOpening":
            text = str(params.get("message"))[:200]
            self.notes.append(f"dismissed a {params.get('type')} dialog: {text}")
            self.send("Page.handleJavaScriptDialog", {"accept": False}, session)

    def _set_up(self, session: str, target: dict) -> None:
        """A new target, paused: no WebSockets, every response's address reported, its own
        new targets attached the same way; then it may run."""
        if target["type"] == "shared_worker":
            return  # stays paused for good: the WebSocket cut-off does not reach shared workers
        if target["type"] in TARGET_TYPES:
            self.send("Network.enable", {}, session)
            rules = {"matchedNetworkConditions": NO_WEBSOCKETS}
            self._guards.add(self.send("Network.emulateNetworkConditionsByRule", rules, session))
            auto = {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True}
            self.send("Target.setAutoAttach", auto, session)
            if target["type"] == "page":
                self.send("Page.enable", {}, session)  # dialogs, in popups too
                self.pages[target["targetId"]] = session
        self.send("Runtime.runIfWaitingForDebugger", {}, session)

    def _allowed(self, url: str) -> bool:
        """The address rules for one request Chrome is about to make (the name is looked up
        each time: a cached answer would widen the window for DNS rebinding)."""
        scheme = url.split(":", 1)[0].lower()
        if scheme in ("data", "blob", "about"):
            return True
        if scheme not in ("http", "https"):
            return False  # file:, chrome: and the like
        try:
            net.check_url(url, allow_private=self.allow_private, timeout=5)
        except net.WebError:
            return False
        return True

    def _address_ok(self, address: str) -> bool:
        """Whether Chrome may have talked to this IP address."""
        try:
            return not net.is_metadata(address) and (self.allow_private or net.is_public(address))
        except ValueError:
            return False

    def js(self, expression: str, timeout: float = COMMAND_SECONDS):
        """The value of a JavaScript expression in the page (a promise is awaited)."""
        result = self.cmd(
            "Runtime.evaluate",
            timeout=timeout,
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            text = details.get("exception", {}).get("description") or details.get("text")
            raise ToolError(f"JavaScript error: {str(text)[:500]}")
        return result.get("result", {}).get("value")

    def wait_load(self) -> None:
        """Until the document is complete (or LOAD_SECONDS), then a moment for late scripts."""
        deadline = time.monotonic() + LOAD_SECONDS
        self.pump(0.3)
        while time.monotonic() < deadline:
            try:
                if self.js("document.readyState", timeout=5) == "complete":
                    break
            except ToolError:
                pass  # "navigated or closed" while a new page replaces the old one
            self.pump(0.2)
        self.pump(0.3)

    def close(self) -> None:
        try:
            self.send("Browser.close", {}, None)
            self.child.proc.wait(5)
        except Exception:
            pass  # killed below either way
        if hasattr(self, "ws"):
            self.ws.close()
        self.child.kill()
        shutil.rmtree(self.scratch, ignore_errors=True)


# --- page actions ---------------------------------------------------------------------------

# Tags each link, button and field with data-arti-ref="e<n>" and returns the page's text with
# those elements as `[e<n>] role "name"` lines (at most MAX characters).
SNAPSHOT_JS = r"""(() => {
const pick = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],' +
  '[role=checkbox],[role=radio],[role=tab],[role=menuitem],[role=option],[role=switch],' +
  '[role=combobox],[contenteditable=""],[contenteditable=true],[onclick]';
const skip = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'svg', 'IFRAME']);
document.querySelectorAll('[data-arti-ref]').forEach(e => e.removeAttribute('data-arti-ref'));
let n = 0, size = 0; const out = [], MAX = 200000;
const add = s => { out.push(s); size += s.length; };
const clean = (s, max = 100) => (s || '').replace(/\s+/g, ' ').trim().slice(0, max);
const shown = e => e.checkVisibility() || getComputedStyle(e).display === 'contents';
const walk = node => { for (const e of node.childNodes) {
  if (size >= MAX) return;
  if (e.nodeType === 3) { const t = clean(e.textContent, 5000); if (t) add(t); continue; }
  if (e.nodeType !== 1 || skip.has(e.tagName) || !shown(e)) continue;
  if (e.matches(pick)) {
    const ref = 'e' + (++n); e.setAttribute('data-arti-ref', ref);
    const tag = e.tagName.toLowerCase();
    const role = e.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'input' ? e.type : tag);
    const img = e.querySelector('img');
    const name = clean(e.getAttribute('aria-label') || e.innerText || e.placeholder ||
      e.title || e.alt || e.name || (img && img.alt));
    let line = `[${ref}] ${role} "${name}"`;
    if ('value' in e && e.value && e.type !== 'password' && tag !== 'button')
      line += ` value="${clean(e.value)}"`;
    if (e.checked) line += ' checked';
    if (tag === 'a') line += ` -> ${e.href.slice(0, 150)}`;
    add(line);
    if (tag !== 'a' && tag !== 'button' && tag !== 'select') walk(e);
    continue;
  }
  if (e.shadowRoot) walk(e.shadowRoot);
  walk(e);
} };
walk(document.body || document.documentElement);
if (size >= MAX) out.push(`[Arti: the page text was cut at ${MAX} characters]`);
return {title: document.title, url: location.href, lines: out};
})()"""

# Finds a snapshot element, scrolls it into view and says where it is and whether it is a
# secret field; with focus, it also focuses it and selects its text (type replaces it).
FIND_JS = r"""((ref, focus) => {
const e = document.querySelector(`[data-arti-ref="${ref}"]`);
if (!e) return null;
const tokens = (e.autocomplete || '').toLowerCase().split(/\s+/);
const secret = e.type === 'password' || tokens.some(t => t.includes('password') ||
  t.startsWith('cc-') || t === 'one-time-code');
e.scrollIntoView({block: 'center', inline: 'center'});
const a = e.closest('a'); if (a) a.target = '_self';  // stay in this tab
if (focus && !secret) {
  e.focus(); if (e.select) e.select(); else getSelection().selectAllChildren(e);
}
const r = e.getBoundingClientRect();
return {x: r.left + r.width / 2, y: r.top + r.height / 2, secret: secret};
})"""


def format_snapshot(page: dict) -> str:
    """Title, URL and the page lines, with runs of plain text lines joined, marked untrusted."""
    lines = []
    for element, group in itertools.groupby(page.get("lines") or [], lambda s: s.startswith("[e")):
        lines += list(group) if element else [" ".join(group)]
    body = "\n".join(lines) or "(the page shows no text)"
    url = str(page.get("url") or "")
    title = str(page.get("title") or "")[:300]
    return f"{net.untrusted(url)}\nTitle: {title}\nURL: {url[:500]}\n\n{body}"


def _find(chrome: Chrome, args: dict, focus: bool = False) -> dict:
    ref = args.get("ref") or ""
    if not re.fullmatch(r"e\d{1,6}", ref):
        raise ToolError("ref must look like e12 (from the latest snapshot)")
    found = chrome.js(f"{FIND_JS}({json.dumps(ref)}, {json.dumps(focus)})")
    if not found:
        raise ToolError(f"no element {ref} on the page now; take a new snapshot")
    return found


def _key(chrome: Chrome, name: str) -> None:
    event = {"key": name, "code": name, "windowsVirtualKeyCode": KEYS[name]}
    text = "\r" if name == "Enter" else ""
    chrome.cmd("Input.dispatchKeyEvent", type="keyDown", text=text, **event)
    chrome.cmd("Input.dispatchKeyEvent", type="keyUp", **event)


def act(chrome: Chrome, args: dict, ctx):
    """Run one action on chrome and build the answer."""
    action = args["action"]
    if action == "navigate":
        url = args.get("url", "").strip()
        try:  # the request check would block it too; this gives the clear reason
            net.check_url(url, allow_private=chrome.allow_private)
        except net.WebError as exc:
            raise ToolError(str(exc)) from None
        error = chrome.cmd("Page.navigate", url=url).get("errorText")
        if error:
            raise ToolError(f"could not open {url}: {error}")
    elif action == "click":
        spot = _find(chrome, args)
        point = {"x": spot["x"], "y": spot["y"]}
        chrome.cmd("Input.dispatchMouseEvent", type="mouseMoved", **point)
        for kind in ("mousePressed", "mouseReleased"):
            chrome.cmd("Input.dispatchMouseEvent", type=kind, button="left", clickCount=1, **point)
    elif action == "type":
        if _find(chrome, args, focus=True)["secret"]:
            raise ToolError(
                "that is a password, card or one-time-code field; Arti never types "
                "those (the owner signs in with browser-login)"
            )
        chrome.cmd("Input.insertText", text=args.get("text", ""))
        if args.get("submit"):
            _key(chrome, "Enter")
    elif action == "press":
        if args.get("key") not in KEYS:
            raise ToolError(f"key must be one of {', '.join(KEYS)}")
        _key(chrome, args["key"])
    elif action == "back":
        chrome.js("history.back()")
    elif action == "scroll":
        dx, dy = SCROLLS[args.get("direction", "down")]
        chrome.js(f"scrollBy(innerWidth * {dx * 0.8}, innerHeight * {dy * 0.8})")
        chrome.pump(0.5)
    elif action == "screenshot":
        shot = chrome.cmd("Page.captureScreenshot", format="png")
        path = ctx.cache_dir("browser") / f"shot-{uuid.uuid4().hex}.png"
        path.write_bytes(base64.b64decode(shot["data"]))
        text = f"Screenshot of {chrome.js('location.href')}"
        return [{"type": "image", "path": str(path)}, {"type": "text", "text": text}]
    elif action == "evaluate":
        text = json.dumps(chrome.js(args.get("js", "")), ensure_ascii=False, default=str)
        if len(text) > MAX_EVALUATE_CHARS:
            text = text[:MAX_EVALUATE_CHARS] + f"\n[Arti: cut at {MAX_EVALUATE_CHARS} characters]"
        return text
    if action in CHANGES_PAGE:
        chrome.wait_load()
    return format_snapshot(chrome.js(SNAPSHOT_JS) or {})


# --- the open browsers, and the tool --------------------------------------------------------

_browsers: dict[tuple[str, str], Chrome] = {}  # (context folder, profile) -> its Chrome
_lock = threading.Lock()  # guards _browsers; held while a Chrome starts: one Chrome per key


def close_idle(older_than: float = IDLE_SECONDS) -> None:
    """Close browsers unused for older_than seconds (and not busy right now)."""
    now, idle = time.monotonic(), []
    with _lock:
        for key, chrome in list(_browsers.items()):
            if now - chrome.last_used >= older_than and chrome.lock.acquire(blocking=False):
                idle.append(_browsers.pop(key))
    for chrome in idle:  # closing takes seconds: outside the lock
        chrome.close()
        chrome.lock.release()


def close_all(app=None) -> None:
    close_idle(older_than=-1)


def _sweeper() -> None:
    """Runs while any browser is open: closes the idle ones."""
    while True:
        time.sleep(15)
        close_idle()
        with _lock:
            if not _browsers:
                return


def _forget(key: tuple[str, str], chrome: Chrome) -> None:
    with _lock:
        if _browsers.get(key) is chrome:
            del _browsers[key]


def _get_browser(ctx, key: tuple[str, str]) -> Chrome:
    profile = key[1]
    if profile and not PROFILE_NAME.fullmatch(profile):
        raise ToolError("profile names are letters, digits, - and _")
    profile_dir = ctx.context_dir / "browser_profiles" / profile
    if profile and not profile_dir.is_dir():
        raise ToolError(
            f"no browser profile {profile!r} in this chat; the owner creates it "
            f"with: python -m arti browser-login {profile}"
        )
    close_idle()
    with _lock:
        chrome = _browsers.get(key)
        if chrome is None:
            if len(_browsers) >= MAX_BROWSERS:
                raise ToolError("too many browsers open; try again in a few minutes")
            scratch = fsutil.private_dir(ctx.cache_dir("browser") / uuid.uuid4().hex)
            if not profile:  # the throw-away profile goes away with the scratch folder
                profile_dir = fsutil.private_dir(scratch / "profile")
            try:
                chrome = Chrome(chrome_path(ctx.app), profile_dir, scratch, net.private_ok(ctx))
            except (OSError, BrowserError, TimeoutError) as exc:
                shutil.rmtree(scratch, ignore_errors=True)
                raise ToolError(f"could not start Chrome: {exc}") from None
            if not _browsers:
                threading.Thread(target=_sweeper, name="browser-idle", daemon=True).start()
            _browsers[key] = chrome
        chrome.last_used = time.monotonic()  # so the idle sweeper leaves it alone
        return chrome


def browser(args: dict, ctx):
    key = (str(ctx.context_dir), args.get("profile", ""))
    if args["action"] == "close":
        with _lock:
            chrome = _browsers.pop(key, None)
        if chrome:
            with chrome.lock:
                chrome.close()
        return "Browser closed."
    chrome = _get_browser(ctx, key)
    while not chrome.lock.acquire(timeout=POLL):  # another call is using this browser
        ctx.check()
    try:
        ctx.check()
        return _run(chrome, key, args, ctx)
    finally:
        chrome.last_used = time.monotonic()
        chrome.lock.release()


def _run(chrome: Chrome, key: tuple[str, str], args: dict, ctx):
    """One action on a browser this call holds; any notes are added to the answer."""
    action = args["action"]
    chrome.notes.clear()
    chrome.cancel = ctx.cancel
    try:
        result = act(chrome, args, ctx)
    except Cancelled:
        if action in ACTS_ON_SITE:
            raise UnknownOutcome(
                f"stopped during {action}; it may or may not have happened"
            ) from None
        raise ToolError("cancelled") from None
    except (BrowserError, TimeoutError, OSError) as exc:
        _forget(key, chrome)  # the connection is in an unknown state: start over next time
        chrome.close()
        if action in ACTS_ON_SITE:
            raise UnknownOutcome(
                f"the browser failed during {action} ({exc}); it may or may not "
                "have happened. The browser was reset."
            ) from None
        raise ToolError(f"browser failed: {exc}. The browser was reset.") from None
    notes = [{"type": "text", "text": f"[Arti: {note}]"} for note in chrome.notes[:10]]
    if not notes:
        return result
    return (result if isinstance(result, list) else [result]) + notes


def review(args: dict) -> str | None:
    """Acting on a signed-in profile needs approval, navigating too (a URL can carry what was
    read off a signed-in page); looking, and the throw-away profile, not, except where the call
    could carry data out (a file read earlier, say): a URL (net.url_carries_data), or a script or
    typed text (net.text_carries_data: long, or with a long or encoded word) that a page could
    send on. Both js and text are checked, whichever the action uses."""
    action = args.get("action")
    if args.get("profile") and action in (*ACTS_ON_SITE, "navigate"):
        return "gui"
    if action == "navigate" and net.url_carries_data(args.get("url") or ""):
        return "url"
    if action in ("evaluate", "type"):
        if any(net.text_carries_data(args.get(name) or "") for name in ("js", "text")):
            return "url"
    return None


def register(api) -> None:
    api.settings("browser", SETTINGS)
    api.tool(
        "browser",
        "Drive a headless Chrome. navigate(url) and the other page actions return a snapshot: "
        "the page text with elements tagged [e12] for click(ref) and type(ref, text, submit). "
        "Also press(key), scroll(direction), back, screenshot, evaluate(js) and close. "
        "profile=NAME uses a profile the owner signed in to (default: a fresh one). Password "
        "and card fields cannot be typed into.",
        {
            "properties": {
                "action": {"type": "string", "enum": ACTIONS},
                "url": {"type": "string"},
                "ref": {"type": "string"},
                "text": {"type": "string", "maxLength": 20_000},
                "submit": {"type": "boolean"},
                "key": {"type": "string", "enum": list(KEYS)},
                "direction": {"type": "string", "enum": list(SCROLLS)},
                "js": {"type": "string", "maxLength": 20_000, "description": "an expression"},
                "profile": {"type": "string"},
            },
            "required": ["action"],
        },
        browser,
        review=review,
        timeout=120,
        available=lambda app: os.access(chrome_path(app), os.X_OK),
    )
    api.on_stop(close_all)
    api.command("browser-login", login_command, "sign in to sites in a browser profile by hand")


# --- the owner's login window ---------------------------------------------------------------


def login_command(app, argv: list[str]) -> int:
    """python -m arti browser-login NAME [--forget] [--scope chat-...]: sign in by hand."""
    parser = argparse.ArgumentParser(prog="python -m arti browser-login")
    parser.add_argument("name")
    parser.add_argument("--forget", action="store_true", help="delete the profile")
    parser.add_argument("--scope", default="home", help="the place (scope) that may use it")
    args = parser.parse_args(argv)
    data_dir = app.data_dir
    place = args.scope == "home" or (data_dir / "contexts" / args.scope).is_dir()  # a known one
    if not (PROFILE_NAME.fullmatch(args.name) and SCOPE.fullmatch(args.scope) and place):
        parser.error("invalid profile name or scope")
    base = data_dir if args.scope == "home" else data_dir / "contexts" / args.scope
    profile = base / "browser_profiles" / args.name
    if args.forget:
        shutil.rmtree(profile, ignore_errors=True)
        print(f"Deleted browser profile {args.name}.")
        return 0
    if not os.access(chrome := chrome_path(app), os.X_OK):
        print("No Chrome or Chromium found; set browser.executable in config.toml.")
        return 1
    os.umask(0o077)
    fsutil.private_dir(profile)
    scratch = fsutil.private_dir(data_dir / "cache" / "browser" / uuid.uuid4().hex)
    child, endpoint = start_chrome(chrome, profile, scratch, headless=False)
    print(f"Sign in to the sites Arti should use in the Chrome window (profile {args.name}).")
    print("Press Enter here when you are done.")
    try:
        sys.stdin.readline()
    finally:
        try:  # a proper quit writes the cookies to disk first
            WebSocket(endpoint).send(json.dumps({"id": 1, "method": "Browser.close"}))
            child.proc.wait(30)
        except Exception:
            pass
        child.kill()
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Saved. The browser tool can use it with profile={args.name!r}.")
    return 0
