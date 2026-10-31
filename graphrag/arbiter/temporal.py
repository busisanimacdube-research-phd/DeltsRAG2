from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

NEG_INF = -10**9
POS_INF = 10**9

_ISO_RE = re.compile(r"^\s*(\d{3,4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s*$")

YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2}|21\d{2})\b")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS.keys()) + r")\.?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(1\d{3}|20\d{2})\b",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS.keys()) + r")\.?,?\s+(1\d{3}|20\d{2})\b",
    re.IGNORECASE,
)


def to_days(ts: Optional[str]) -> Optional[int]:
    if ts is None:
        return None
    s = str(ts).strip()
    if not s or s.lower() in ("static", "unknown", "none", "0", "present", "n/a"):
        return None
    m = _ISO_RE.match(s)
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else 1
    day = int(m.group(3)) if m.group(3) else 1
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    return year * 365 + month * 30 + day


def normalise_timestamp(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    m = _ISO_RE.match(s)
    if not m:
        return None
    year = int(m.group(1))
    if m.group(2) is None:
        return f"{year:04d}"
    month = int(m.group(2))
    if not (1 <= month <= 12):
        return f"{year:04d}"
    if m.group(3) is None:
        return f"{year:04d}-{month:02d}"
    day = int(m.group(3))
    if not (1 <= day <= 31):
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_date_expression(text: str) -> Optional[str]:
    if not text:
        return None

    m = _DAY_MONTH_YEAR_RE.search(text)
    if m:
        day, mon, year = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
        return f"{year:04d}-{mon:02d}-{min(day, 31):02d}"

    m = _MONTH_YEAR_RE.search(text)
    if m:
        mon = _MONTHS[m.group(1).lower()]
        year = int(m.group(3))
        if m.group(2):
            return f"{year:04d}-{mon:02d}-{min(int(m.group(2)), 31):02d}"
        return f"{year:04d}-{mon:02d}"

    m = YEAR_RE.search(text)
    if m:
        return m.group(1)

    return None


@dataclass(frozen=True)
class Interval:

    start: Optional[str] = None
    end: Optional[str] = None
    start_inferred: bool = False
    end_inferred: bool = False

    @property
    def start_days(self) -> int:
        d = to_days(self.start)
        return NEG_INF if d is None else d

    @property
    def end_days(self) -> int:
        d = to_days(self.end)
        return POS_INF if d is None else d

    @property
    def is_open_ended(self) -> bool:
        return to_days(self.end) is None

    @property
    def is_point(self) -> bool:
        return self.start is not None and self.start == self.end

    def contains(self, day: int) -> bool:
        return self.start_days <= day <= self.end_days

    def overlaps(self, other: "Interval") -> bool:
        return self.start_days <= other.end_days and other.start_days <= self.end_days

    def distance_to(self, day: int) -> int:
        if self.contains(day):
            return 0
        if day < self.start_days:
            return self.start_days - day
        return day - self.end_days

    def with_end(self, end: Optional[str], inferred: bool = True) -> "Interval":
        return Interval(
            start=self.start,
            end=end,
            start_inferred=self.start_inferred,
            end_inferred=inferred,
        )

    def render(self) -> str:
        lo = self.start or "?"
        hi = self.end if self.end else "open"
        lo_mark = "~" if self.start_inferred else ""
        hi_mark = "~" if self.end_inferred else ""
        return f"[{lo_mark}{lo} → {hi_mark}{hi}]"


def parse_interval_expression(text: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    if not text:
        return None

    lowered = text.lower()

    m = re.search(
        r"\bfrom\s+(.{1,40}?)\s+(?:to|until|till|through|up to|–|—|-)\s+(.{1,40}?)(?:[,.;)]|\s+(?:and|where|when|while|as)\b|$)",
        lowered,
    )
    if m:
        start = parse_date_expression(m.group(1))
        end_text = m.group(2)
        if re.search(r"\b(present|now|today|date)\b", end_text):
            return (start, None)
        end = parse_date_expression(end_text)
        if start or end:
            return (start, end)

    m = re.search(r"\bbetween\s+(.{1,40}?)\s+and\s+(.{1,40}?)(?:[,.;)]|$)", lowered)
    if m:
        start = parse_date_expression(m.group(1))
        end = parse_date_expression(m.group(2))
        if start and end:
            return (start, end)

    m = re.search(r"\b(1\d{3}|20\d{2})\s*(?:–|—|--|-)\s*(1\d{3}|20\d{2})\b", text)
    if m:
        return (m.group(1), m.group(2))

    m = re.search(r"\b(1\d{3}|20\d{2})\s*(?:–|—|--|-)\s*(?:present|now|date)\b", lowered)
    if m:
        return (m.group(1), None)

    m = re.search(r"\b(?:since|starting (?:in|from)|as of|beginning in|effective)\s+(.{1,30}?)(?:[,.;)]|$)", lowered)
    if m:
        start = parse_date_expression(m.group(1))
        if start:
            return (start, None)

    m = re.search(r"\b(?:until|till|up to|through)\s+(.{1,30}?)(?:[,.;)]|$)", lowered)
    if m:
        end = parse_date_expression(m.group(1))
        if end:
            return (None, end)

    return None
