from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .assertions import Assertion
from .temporal import to_days
from .timeline import StateTimeline

logger = logging.getLogger("DELTA-RAG")


STATUS_HOLDS = "holds"
STATUS_SUPERSEDED = "superseded"
STATUS_NOT_YET = "not_yet_effective"
STATUS_ALTERNATIVE = "alternative"
STATUS_UNDATED = "undated"
STATUS_LATEST = "latest_known"


@dataclass
class ACAConfig:

    w_containment: float = 0.50
    w_proximity: float = 0.20
    w_recency: float = 0.15
    w_corroboration: float = 0.15

    proximity_decay: float = 0.0015
    open_ended_penalty: float = 0.6
    min_holder_score: float = 0.25
    corroboration_saturation: int = 4
    enable_corroboration: bool = True

    def effective_weights(self):
        if self.enable_corroboration:
            return (self.w_containment, self.w_proximity,
                    self.w_recency, self.w_corroboration)
        rest = self.w_containment + self.w_proximity + self.w_recency
        if rest <= 0:
            return (1.0, 0.0, 0.0, 0.0)
        scale = 1.0 / rest
        return (self.w_containment * scale, self.w_proximity * scale,
                self.w_recency * scale, 0.0)


@dataclass
class ArbitrationResult:

    timeline: StateTimeline
    anchor: Optional[str]
    holder: Optional[Assertion]
    scores: Dict[int, float]
    decided: bool

    @property
    def slot_label(self) -> str:
        return self.timeline.slot_label


class AnchorConditionedArbitrator:

    def __init__(self, config: Optional[ACAConfig] = None):
        self.config = config or ACAConfig()

    def arbitrate(
        self,
        timelines: Sequence[StateTimeline],
        anchor: Optional[str],
    ) -> List[ArbitrationResult]:
        anchor_days = to_days(anchor)
        results = [self._arbitrate_one(tl, anchor, anchor_days) for tl in timelines]
        n_decided = sum(1 for r in results if r.decided)
        logger.info(
            f"ARBITER/ACA: decided {n_decided}/{len(results)} slots at anchor={anchor}"
        )
        return results

    def _arbitrate_one(
        self,
        timeline: StateTimeline,
        anchor: Optional[str],
        anchor_days: Optional[int],
    ) -> ArbitrationResult:
        assertions = timeline.assertions
        if not assertions:
            return ArbitrationResult(timeline, anchor, None, {}, False)

        if anchor_days is None:
            return self._resolve_without_anchor(timeline, anchor)

        max_corr = max((a.corroboration for a in assertions), default=1)
        scores: Dict[int, float] = {}
        for i, a in enumerate(assertions):
            scores[i] = self._score(a, anchor_days, assertions, max_corr)

        best_idx = max(scores, key=lambda i: scores[i])
        best_score = scores[best_idx]
        decided = best_score >= self.config.min_holder_score

        for i, a in enumerate(assertions):
            a.status = self._status_for(a, anchor_days, decided and i == best_idx)

        holder = assertions[best_idx] if decided else None
        return ArbitrationResult(timeline, anchor, holder, scores, decided)

    def _resolve_without_anchor(
        self, timeline: StateTimeline, anchor: Optional[str]
    ) -> ArbitrationResult:
        dated = [
            (i, a) for i, a in enumerate(timeline.assertions)
            if to_days(a.interval.start) is not None
        ]
        for a in timeline.assertions:
            a.status = STATUS_UNDATED
        if not dated:
            return ArbitrationResult(timeline, anchor, None, {}, False)

        latest_idx, latest = max(dated, key=lambda t: to_days(t[1].interval.start) or 0)
        for a in timeline.assertions:
            if to_days(a.interval.start) is not None:
                a.status = STATUS_SUPERSEDED
        latest.status = STATUS_LATEST
        return ArbitrationResult(
            timeline, anchor, latest, {latest_idx: 1.0}, True
        )


    def _score(
        self,
        a: Assertion,
        anchor_days: int,
        siblings: Sequence[Assertion],
        max_corr: int,
    ) -> float:
        cfg = self.config
        iv = a.interval

        start = to_days(iv.start)
        end = to_days(iv.end)
        if start is None and end is None:
            return 0.0

        contains = iv.contains(anchor_days)
        containment = 1.0 if contains else 0.0
        if contains and iv.is_open_ended:
            containment *= cfg.open_ended_penalty

        distance = iv.distance_to(anchor_days)
        proximity = math.exp(-cfg.proximity_decay * distance)

        recency = 0.0
        if start is not None and start <= anchor_days:
            starts = [
                s for s in (to_days(x.interval.start) for x in siblings)
                if s is not None and s <= anchor_days
            ]
            if starts:
                lo, hi = min(starts), max(starts)
                recency = 1.0 if hi == lo else (start - lo) / (hi - lo)

        corroboration = 0.0
        if max_corr > 0:
            sat = max(cfg.corroboration_saturation, 1)
            corroboration = math.log1p(min(a.corroboration, sat)) / math.log1p(sat)

        w_con, w_prox, w_rec, w_corr = cfg.effective_weights()
        return (
            w_con * containment
            + w_prox * proximity
            + w_rec * recency
            + w_corr * corroboration
        )

    @staticmethod
    def _status_for(a: Assertion, anchor_days: int, is_holder: bool) -> str:
        if is_holder:
            return STATUS_HOLDS
        start = to_days(a.interval.start)
        end = to_days(a.interval.end)
        if start is None and end is None:
            return STATUS_UNDATED
        if end is not None and end < anchor_days:
            return STATUS_SUPERSEDED
        if start is not None and start > anchor_days:
            return STATUS_NOT_YET
        return STATUS_ALTERNATIVE
