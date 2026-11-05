"""Schedule strings for cron jobs: parsing and "when does it fire next". Pure: no I/O.

A parsed schedule is a small JSON-safe dict, stored in the job:
  {"kind": "once", "at": "<ISO time>"}        "in 30m", "once at 2026-10-24T09:00", a bare ISO time
  {"kind": "every", "minutes": n}             "every 2h", a bare "30m"
  {"kind": "cron", "expr": "10 7 * * *"}      minute hour day-of-month month day-of-week
Wall-clock times are read in the time zone of the aware `now`/`after` the caller passes. Daylight
saving uses plain zoneinfo: a time a spring-forward skips (02:30) fires at that time an hour
later (03:30), a repeated hour fires once, and intervals are elapsed time.
"""

import re
from datetime import datetime, timedelta, timezone

_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_DAYS = "sun mon tue wed thu fri sat".split()
# Per field (minute, hour, day of month, month, day of week): allowed range and names.
_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
_NAMES = [{}, {}, {}, {n: i for i, n in enumerate(_MONTHS, 1)}, {n: i for i, n in enumerate(_DAYS)}]
_UNITS = {
    **dict.fromkeys(["m", "min", "mins", "minute", "minutes"], 1),
    **dict.fromkeys(["h", "hr", "hrs", "hour", "hours"], 60),
    **dict.fromkeys(["d", "day", "days"], 1440),
}
_DURATION = re.compile(r"(\d+)\s*([a-z]+)")
_SEARCH_DAYS = 366 * 5  # Feb 29 on a given weekday can be years away


def parse(text: str, now: datetime) -> dict:
    """The schedule dict for `text`; ValueError with a hint when it is not one."""
    text = " ".join(text.strip().lower().split())
    if text.startswith("in "):
        return {"kind": "once", "at": _later(now, _minutes(text[3:])).isoformat()}
    if text.startswith("every "):
        return {"kind": "every", "minutes": _minutes(text[6:])}
    if _DURATION.fullmatch(text):
        return {"kind": "every", "minutes": _minutes(text)}
    if len(text.split()) == 5:
        _fields(text)  # raises on a bad expression
        return {"kind": "cron", "expr": text}
    iso = text.removeprefix("once at ").upper()
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(
            f"can't read schedule {text!r}; use 'in 30m', 'every 2h', 'once at <ISO time>' "
            "or a cron expression like '10 7 * * *'"
        ) from None
    if when.tzinfo is None:
        when = when.replace(tzinfo=now.tzinfo)
    return {"kind": "once", "at": when.astimezone(now.tzinfo).isoformat()}


def next_run(sched: dict, after: datetime) -> datetime | None:
    """The first fire time strictly after `after`; None when there is none (a past one-shot)."""
    if sched["kind"] == "once":
        at = datetime.fromisoformat(sched["at"])
        return at if at > after else None
    if sched["kind"] == "every":
        return _later(after, sched["minutes"])
    return _cron_next(sched["expr"], after)


def period(sched: dict, now: datetime) -> timedelta | None:
    """How often it runs around `now`: the gap between its next two fires (None for a one-shot)."""
    if sched["kind"] == "once":
        return None
    first = next_run(sched, now)
    return next_run(sched, first) - first


def shortest_gap(sched: dict, now: datetime, fires: int = 50) -> timedelta | None:
    """The shortest gap between its next `fires` fires (None for a one-shot): irregular schedules
    ("0,30 9 * * *") count by their closest pair, so a minimum gap holds for every fire. A rare
    schedule (Feb 29) stops at the search horizon instead of failing."""
    if sched["kind"] == "once":
        return None
    times = [next_run(sched, now)]
    for _ in range(fires):
        try:
            times.append(next_run(sched, times[-1]))
        except ValueError:
            break
    return min((b - a for a, b in zip(times, times[1:])), default=None)


def _later(when: datetime, minutes: int) -> datetime:
    """`when` plus elapsed minutes (aware arithmetic in one zone would add wall-clock time)."""
    return (when.astimezone(timezone.utc) + timedelta(minutes=minutes)).astimezone(when.tzinfo)


def _minutes(text: str) -> int:
    match = _DURATION.fullmatch(text.strip())
    if not match or match[2] not in _UNITS or int(match[1]) < 1:
        raise ValueError(f"can't read duration {text!r}; use e.g. 30m, 2h or 1d")
    return int(match[1]) * _UNITS[match[2]]


def _fields(expr: str) -> list[set[int]]:
    """The allowed values of each of the five fields."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("a cron expression has 5 fields: minute hour day month weekday")
    fields = []
    for index, part in enumerate(parts):
        values = set()
        for piece in part.split(","):
            values.update(_piece(piece, index))
        fields.append(values)
    if 7 in fields[4]:  # 0 and 7 both mean Sunday
        fields[4] = (fields[4] - {7}) | {0}
    return fields


def _piece(piece: str, index: int) -> range:
    """The values of one list item: `*`, `5`, `1-5` or `mon-fri`, each optionally `/step`."""
    low, high = _RANGES[index]
    base, slash, step = piece.partition("/")

    def number(token: str) -> int:
        if token in _NAMES[index]:
            return _NAMES[index][token]
        if not token.isdigit():
            raise ValueError(f"bad cron value {token!r}")
        return int(token)

    if base == "*":
        first, last = low, high
    elif "-" in base:
        first, last = (number(t) for t in base.split("-", 1))
    else:
        first = number(base)
        last = high if slash else first
    if slash and (not step.isdigit() or int(step) < 1):
        raise ValueError(f"bad cron step in {piece!r}")
    if not low <= first <= last <= high:
        raise ValueError(f"cron value out of range in {piece!r}")
    return range(first, last + 1, int(step) if slash else 1)


def _cron_next(expr: str, after: datetime) -> datetime:
    minutes, hours, days, months, weekdays = _fields(expr)
    parts = expr.split()
    any_day, any_weekday = parts[2].startswith("*"), parts[4].startswith("*")
    day = after.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    for _ in range(_SEARCH_DAYS):
        day_ok, weekday_ok = day.day in days, (day.weekday() + 1) % 7 in weekdays
        # Standard cron: when both day fields are restricted, either one matching is enough.
        day_matches = day_ok and weekday_ok if any_day or any_weekday else day_ok or weekday_ok
        if day.month in months and day_matches:
            for hour in sorted(hours):
                for minute in sorted(minutes):
                    candidate = day.replace(hour=hour, minute=minute, tzinfo=after.tzinfo)
                    # Compare instants, so a skipped (DST) wall time still orders correctly.
                    if candidate.timestamp() > after.timestamp():
                        return candidate
        day += timedelta(days=1)
    raise ValueError(f"cron expression {expr!r} never fires")
