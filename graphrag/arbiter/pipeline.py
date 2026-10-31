from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .arbitration import ACAConfig, AnchorConditionedArbitrator, STATUS_HOLDS
from .assertions import TANConfig, TypedAssertionNormaliser
from .rendering import ACRConfig, ArbitratedContextRenderer
from .temporal import to_days
from .timeline import STCConfig, StateTimelineConsolidator

logger = logging.getLogger("DELTA-RAG")


@dataclass
class ARBITERConfig:

    tan: TANConfig = field(default_factory=TANConfig)
    stc: STCConfig = field(default_factory=STCConfig)
    aca: ACAConfig = field(default_factory=ACAConfig)
    acr: ACRConfig = field(default_factory=ACRConfig)

    require_slot: bool = True
    require_conflict: bool = False


@dataclass
class ARBITERStats:

    n_events: int = 0
    n_assertions: int = 0
    n_unslotted: int = 0
    n_slots: int = 0
    n_conflicted_slots: int = 0
    n_multivalue_slots: int = 0
    n_decided_slots: int = 0
    n_intervals_recovered: int = 0
    n_supersession_closures: int = 0
    n_terminator_closures: int = 0
    anchor: Optional[str] = None
    applied: bool = False

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


class ARBITER:

    def __init__(self, config: Optional[ARBITERConfig] = None):
        self.config = config or ARBITERConfig()
        self.normaliser = TypedAssertionNormaliser(self.config.tan)
        self.consolidator = StateTimelineConsolidator(self.config.stc)
        self.arbitrator = AnchorConditionedArbitrator(self.config.aca)
        self.renderer = ArbitratedContextRenderer(self.config.acr)
        self._last_stats = ARBITERStats()

    @staticmethod
    def resolve_anchor(time_constraints: Optional[Dict]) -> Optional[str]:
        if not time_constraints:
            return None
        return time_constraints.get("start_time") or time_constraints.get("end_time")

    def arbitrate(
        self,
        events: Sequence[Dict],
        query_entities: Sequence[str],
        time_constraints: Optional[Dict] = None,
    ) -> Tuple[Optional[str], ARBITERStats]:
        stats = ARBITERStats(n_events=len(events))
        anchor = self.resolve_anchor(time_constraints)
        stats.anchor = anchor

        if not events:
            self._last_stats = stats
            return None, stats

        assertions, unslotted = self.normaliser.normalise(events, query_entities)
        stats.n_assertions = len(assertions)
        stats.n_unslotted = len(unslotted)
        stats.n_intervals_recovered = sum(
            1 for a in assertions if not a.interval.is_open_ended
        )

        if self.config.require_slot and not assertions:
            logger.info("ARBITER: no slots typed; falling back to chronological context")
            self._last_stats = stats
            return None, stats

        timelines = self.consolidator.consolidate(assertions)
        stats.n_slots = len(timelines)
        stats.n_conflicted_slots = sum(1 for tl in timelines if tl.has_conflict)
        stats.n_multivalue_slots = sum(
            1 for tl in timelines if tl.n_distinct_values > 1)
        closures = [a.closure for tl in timelines for a in tl.assertions]
        stats.n_supersession_closures = sum(1 for c in closures if c == "supersession")
        stats.n_terminator_closures = sum(1 for c in closures if c == "terminator")

        if self.config.require_conflict and stats.n_conflicted_slots == 0:
            logger.info("ARBITER: no slot conflicts; falling back to chronological context")
            self._last_stats = stats
            return None, stats

        results = self.arbitrator.arbitrate(timelines, anchor)
        stats.n_decided_slots = sum(1 for r in results if r.decided)

        block = self.renderer.render(results, unslotted, anchor)
        if not block or not block.strip():
            self._last_stats = stats
            return None, stats

        stats.applied = True
        self._last_stats = stats
        logger.info(
            f"ARBITER: applied — {stats.n_slots} slots "
            f"({stats.n_multivalue_slots} multi-value, "
            f"{stats.n_conflicted_slots} unresolved-conflict, "
            f"{stats.n_decided_slots} decided), "
            f"{stats.n_supersession_closures} supersession closures, anchor={anchor}"
        )
        return block, stats

    @property
    def last_stats(self) -> ARBITERStats:
        return self._last_stats
