"""The offline test suite: `.venv/bin/python -m unittest discover -s tests -t .`

Importing this package guards the test process: an audit hook makes any connection to, or name
lookup of, a host other than loopback fail the test. ARTI_HOME and every temporary file live
in one fresh directory that is removed at exit, so no test can touch the real ~/.arti.
"""

import atexit
import ipaddress
import os
import shutil
import sys
import tempfile

_CONNECTS = ("socket.connect", "socket.sendto", "socket.sendmsg")
_LOOKUPS = ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr")


def _is_loopback(host) -> bool:
    if host in (None, "localhost", b"localhost"):
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _deny_external(event: str, args: tuple) -> None:
    if event in _CONNECTS:
        address = args[1]
        if isinstance(address, tuple) and not _is_loopback(address[0]):
            raise AssertionError(f"offline tests: no network access to {address[0]!r}")
    elif event in _LOOKUPS and not _is_loopback(args[0]):
        raise AssertionError(f"offline tests: no name lookup of {args[0]!r}")


sys.addaudithook(_deny_external)

# Every temporary file of the run lives in one directory, removed at exit, so tests that call
# tempfile.mkdtemp() without cleaning up leave nothing behind.
tempfile.tempdir = tempfile.mkdtemp(prefix="arti-tests-")
atexit.register(shutil.rmtree, tempfile.tempdir, ignore_errors=True)
os.environ["ARTI_HOME"] = tempfile.mkdtemp(prefix="home-")
