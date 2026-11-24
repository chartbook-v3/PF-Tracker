"""NYSE (XNYS) trading calendar and US Eastern Time market-close helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

ET = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _get_nyse_calendar() -> xcals.ExchangeCalendar:
    """Return cached NYSE (XNYS) exchange calendar instance."""
    return xcals.get_calendar("XNYS")


def _as_date(d: date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    return d


def _normalize_et(now_et: datetime | None) -> datetime:
    """Return a timezone-aware datetime in America/New_York."""
    if now_et is None:
        return datetime.now(ET)
    if now_et.tzinfo is None:
        return now_et.replace(tzinfo=ET)
    return now_et.astimezone(ET)


def is_trading_day(d: date) -> bool:
    """Return True if date `d` is an active NYSE trading session."""
    cal = _get_nyse_calendar()
    target = _as_date(d)
    return bool(cal.is_session(target.isoformat()))


def previous_trading_day(d: date) -> date:
    """Return the most recent NYSE trading day strictly before `d`."""
    cal = _get_nyse_calendar()
    prev_cal_day = _as_date(d) - timedelta(days=1)
    session = cal.date_to_session(prev_cal_day.isoformat(), direction="previous")
    return session.date()


def get_market_close_et(d: date) -> datetime | None:
    """Return exact NYSE market close datetime in ET for `d`, or None if closed.

    Handles standard 16:00 ET closes as well as early closes (e.g. 13:00 ET on
    Black Friday and Christmas Eve).
    """
    target = _as_date(d)
    if not is_trading_day(target):
        return None
    cal = _get_nyse_calendar()
    close_utc = cal.session_close(target.isoformat())
    return close_utc.tz_convert(ET).to_pydatetime()


def is_after_market_close_et(
    d: date,
    now_et: datetime | None = None,
    buffer_minutes: int = 15,
) -> bool:
    """Return True if `now_et` is at least `buffer_minutes` past `d`'s NYSE close."""
    close_et = get_market_close_et(d)
    if close_et is None:
        return False
    current_et = _normalize_et(now_et)
    return current_et >= close_et + timedelta(minutes=buffer_minutes)


def latest_completed_trading_day(
    now_et: datetime | None = None,
    buffer_minutes: int = 15,
) -> date:
    """Return the latest NYSE trading day whose close (+ buffer) has already passed."""
    current_et = _normalize_et(now_et)
    today = current_et.date()
    if is_trading_day(today) and is_after_market_close_et(
        today, now_et=current_et, buffer_minutes=buffer_minutes
    ):
        return today
    return previous_trading_day(today)


def trading_days_between(start_date: date, end_date: date) -> list[date]:
    """Return sorted list of NYSE trading days in [start_date, end_date] inclusive."""
    start = _as_date(start_date)
    end = _as_date(end_date)
    if end < start:
        return []
    cal = _get_nyse_calendar()
    sessions = cal.sessions_in_range(start.isoformat(), end.isoformat())
    return [ts.date() for ts in sessions]
