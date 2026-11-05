import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from plugins.cron import schedule

TZ = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 10, 23, 8, 0, tzinfo=TZ)  # a Friday


class ParseTest(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(schedule.parse("in 30m", NOW)["at"], "2026-10-23T08:30:00-07:00")
        self.assertEqual(schedule.parse("every 2 hours", NOW), {"kind": "every", "minutes": 120})
        self.assertEqual(schedule.parse("30m", NOW), {"kind": "every", "minutes": 30})
        self.assertEqual(schedule.parse("10 7 * * *", NOW), {"kind": "cron", "expr": "10 7 * * *"})
        once = schedule.parse("once at 2026-10-24T09:00", NOW)
        self.assertEqual(once, {"kind": "once", "at": "2026-10-24T09:00:00-07:00"})
        self.assertEqual(schedule.parse("2026-10-24 16:00:00+00:00", NOW)["at"], once["at"])

    def test_errors(self):
        for bad in ("tomorrow", "every 0m", "in 5 parsecs", "61 * * * *", "* * * * fri-mon"):
            with self.assertRaises(ValueError, msg=bad):
                schedule.parse(bad, NOW)


class NextRunTest(unittest.TestCase):
    def next(self, expr, after=NOW):
        return schedule.next_run({"kind": "cron", "expr": expr}, after)

    def test_cron_basics(self):
        self.assertEqual(self.next("10 7 * * *"), datetime(2026, 10, 24, 7, 10, tzinfo=TZ))
        self.assertEqual(self.next("*/5 * * * *"), datetime(2026, 10, 23, 8, 5, tzinfo=TZ))
        self.assertEqual(self.next("0 9 * * mon-fri"), datetime(2026, 10, 23, 9, 0, tzinfo=TZ))
        self.assertEqual(self.next("0 7 * * 7"), datetime(2026, 10, 25, 7, 0, tzinfo=TZ))
        self.assertEqual(self.next("0 0 1 jan *"), datetime(2027, 1, 1, 0, 0, tzinfo=TZ))

    def test_day_fields_or_when_both_restricted(self):
        # The 1st of the month OR a Monday, whichever is first.
        self.assertEqual(self.next("0 12 1 * mon"), datetime(2026, 10, 26, 12, 0, tzinfo=TZ))

    def test_dst(self):
        # Spring forward 2027-03-14: 02:30 does not exist; it fires at the same instant as 03:30.
        fired = self.next("30 2 * * *", datetime(2027, 3, 14, 0, 0, tzinfo=TZ))
        self.assertEqual(fired.timestamp(), datetime(2027, 3, 14, 3, 30, tzinfo=TZ).timestamp())
        # Fall back 2026-11-01: 01:30 happens twice; a daily job fires once.
        first = self.next("30 1 * * *", datetime(2026, 11, 1, 0, 0, tzinfo=TZ))
        self.assertEqual(self.next("30 1 * * *", first).day, 2)

    def test_intervals_are_elapsed_time(self):
        before = datetime(2026, 11, 1, 0, 30, tzinfo=TZ)
        after = schedule.next_run({"kind": "every", "minutes": 120}, before)
        self.assertEqual(after.timestamp() - before.timestamp(), 7200)

    def test_one_shot_and_shortest_gap(self):
        once = schedule.parse("in 5m", NOW)
        self.assertIsNotNone(schedule.next_run(once, NOW))
        self.assertIsNone(schedule.next_run(once, NOW + timedelta(minutes=5)))
        self.assertIsNone(schedule.shortest_gap(once, NOW))
        daily = schedule.parse("10 7 * * *", NOW)
        self.assertEqual(schedule.shortest_gap(daily, NOW), timedelta(days=1))
        twice = schedule.parse("0,30 9 * * *", NOW)  # 30 minutes apart, then a day
        self.assertEqual(schedule.shortest_gap(twice, NOW), timedelta(minutes=30))
        self.assertEqual(schedule.period(daily, NOW), timedelta(days=1))
        leap = schedule.parse("0 9 29 2 *", NOW)  # rare: years apart, never an error
        self.assertGreater(schedule.shortest_gap(leap, NOW), timedelta(days=365))
        self.assertGreater(schedule.period(leap, NOW), timedelta(days=365))
