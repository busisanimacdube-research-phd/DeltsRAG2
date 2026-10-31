from __future__ import annotations
import os
import calendar
from datetime import date
from typing import Optional

_MODE = os.getenv("DELTA_DAY_INDEX", "exact").lower()


def _components(ts) -> Optional[tuple]:
    if ts is None:
        return None
    ts = str(ts).strip()
    if not ts or ts in ("static", "0"):
        return None
    parts = ts.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    except (ValueError, IndexError):
        return None
    return year, month, day


def day_index_approx(ts) -> Optional[int]:
    c = _components(ts)
    if c is None:
        return None
    y, m, d = c
    return y * 365 + m * 30 + d


def day_index_exact(ts) -> Optional[int]:
    c = _components(ts)
    if c is None:
        return None
    y, m, d = c
    m = min(max(m, 1), 12)
    if 1 <= y <= 9999:
        d = min(max(d, 1), calendar.monthrange(y, m)[1])
        return date(y, m, d).toordinal()
    return int(round((y - 1) * 365.2425 + (m - 1) * 30.436875 + d))


def day_index(ts) -> Optional[int]:
    return day_index_approx(ts) if _MODE == "approx" else day_index_exact(ts)
