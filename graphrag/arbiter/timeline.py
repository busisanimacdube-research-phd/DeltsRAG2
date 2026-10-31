from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .assertions import Assertion
from .temporal import Interval, POS_INF, to_days

logger = logging.getLogger("DELTA-RAG")


@dataclass
class STCConfig:

    enable_supersession_closure: bool = True
    enable_terminator_closure: bool = True
    min_assertions_per_slot: int = 1
    merge_identical_values: bool = True


@dataclass
class StateTimeline:

    subject: str
    relation: str
    functional: bool
    assertions: List[Assertion] = field(default_factory=list)
    conflicts: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def slot_label(self) -> str:
        return f"{self.subject} — {self.relation}"

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def n_distinct_values(self) -> int:
        return len({a.value_key for a in self.assertions})


class StateTimelineConsolidator:

    def __init__(self, config: Optional[STCConfig] = None):
        self.config = config or STCConfig()

    def consolidate(self, assertions: Sequence[Assertion]) -> List[StateTimeline]:
        by_slot: Dict[Tuple[str, str], List[Assertion]] = {}
        for a in assertions:
            by_slot.setdefault(a.slot, []).append(a)

        timelines: List[StateTimeline] = []
        for (subj_key, relation), group in by_slot.items():
            merged = self._merge_duplicates(group)
            if len(merged) < self.config.min_assertions_per_slot:
                continue

            functional = any(a.functional for a in merged)
            merged.sort(key=self._sort_key)

            if functional:
                if self.config.enable_terminator_closure:
                    merged = self._apply_terminators(merged)
                if self.config.enable_supersession_closure:
                    merged = self._close_by_supersession(merged)

            timeline = StateTimeline(
                subject=group[0].subject,
                relation=relation,
                functional=functional,
                assertions=merged,
            )
            if functional:
                timeline.conflicts = self._detect_conflicts(merged)
            timelines.append(timeline)

        n_conflicted = sum(1 for t in timelines if t.has_conflict)
        logger.info(
            f"ARBITER/STC: {len(timelines)} slot timelines, "
            f"{n_conflicted} carrying mutual-exclusion conflicts"
        )
        return timelines


    @staticmethod
    def _sort_key(a: Assertion) -> Tuple[int, int]:
        s = to_days(a.interval.start)
        e = to_days(a.interval.end)
        if s is None and e is None:
            return (1, 0)
        if s is None:
            return (0, e)
        return (0, s)

    def _merge_duplicates(self, group: Sequence[Assertion]) -> List[Assertion]:
        if not self.config.merge_identical_values:
            return list(group)

        buckets: Dict[str, List[Assertion]] = {}
        for a in group:
            buckets.setdefault(a.value_key, []).append(a)

        merged: List[Assertion] = []
        for _, items in buckets.items():
            if len(items) == 1:
                items[0].corroboration = 1 if items[0].source_id else 1
                merged.append(items[0])
                continue

            def informativeness(a: Assertion) -> Tuple[int, int]:
                closed = 0 if a.interval.is_open_ended else -1
                start = to_days(a.interval.start)
                return (closed, start if start is not None else POS_INF)

            items.sort(key=informativeness)
            primary = items[0]

            start_owner = min(
                (a for a in items if to_days(a.interval.start) is not None),
                key=lambda a: to_days(a.interval.start),
                default=None,
            )
            end_owner = max(
                (a for a in items if to_days(a.interval.end) is not None),
                key=lambda a: to_days(a.interval.end),
                default=None,
            )

            src_start = start_owner.interval if start_owner else primary.interval
            src_end = end_owner.interval if end_owner else primary.interval

            primary.interval = Interval(
                start=src_start.start,
                end=src_end.end,
                start_inferred=src_start.start_inferred,
                end_inferred=src_end.end_inferred,
            )
            if end_owner is not None and end_owner.aspect == "terminate":
                primary.closure = "terminator"

            primary.corroboration = len({a.source_id for a in items if a.source_id}) or 1

            if any(a.aspect == "initiate" for a in items):
                primary.aspect = "initiate"
            elif any(a.aspect == "terminate" for a in items):
                primary.aspect = "terminate"
            merged.append(primary)

        return merged


    def _apply_terminators(self, assertions: List[Assertion]) -> List[Assertion]:
        terminators = [
            a for a in assertions
            if a.aspect == "terminate" and to_days(a.interval.end) is not None
        ]
        if not terminators:
            return assertions

        for a in assertions:
            if a.aspect == "terminate" or not a.interval.is_open_ended:
                continue
            a_start = to_days(a.interval.start)
            best: Optional[int] = None
            best_str: Optional[str] = None
            for t in terminators:
                if t.value_key != a.value_key:
                    continue
                t_end = to_days(t.interval.end)
                if t_end is None:
                    continue
                if a_start is not None and t_end < a_start:
                    continue
                if best is None or t_end < best:
                    best, best_str = t_end, t.interval.end
            if best_str is not None:
                a.interval = a.interval.with_end(best_str, inferred=False)
                a.closure = "terminator"
        return assertions

    def _close_by_supersession(self, assertions: List[Assertion]) -> List[Assertion]:
        dated = [a for a in assertions if to_days(a.interval.start) is not None]
        dated.sort(key=lambda a: to_days(a.interval.start) or 0)

        for i, a in enumerate(dated):
            if not a.interval.is_open_ended:
                continue
            a_start = to_days(a.interval.start)
            successor_start: Optional[str] = None
            for b in dated[i + 1:]:
                if b.value_key == a.value_key:
                    continue
                b_start = to_days(b.interval.start)
                if b_start is None or a_start is None or b_start <= a_start:
                    continue
                successor_start = b.interval.start
                break
            if successor_start is not None:
                a.interval = a.interval.with_end(successor_start, inferred=True)
                a.superseded_by = successor_start
                a.closure = "supersession"
        return assertions


    @staticmethod
    def _detect_conflicts(assertions: Sequence[Assertion]) -> List[Tuple[int, int]]:
        conflicts: List[Tuple[int, int]] = []
        for i in range(len(assertions)):
            for j in range(i + 1, len(assertions)):
                a, b = assertions[i], assertions[j]
                if a.value_key == b.value_key:
                    continue
                if a.interval.overlaps(b.interval):
                    conflicts.append((i, j))
        return conflicts
