from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .arbitration import (
    ArbitrationResult,
    STATUS_ALTERNATIVE,
    STATUS_HOLDS,
    STATUS_LATEST,
    STATUS_NOT_YET,
    STATUS_SUPERSEDED,
    STATUS_UNDATED,
)
from .assertions import Assertion
from .temporal import to_days

logger = logging.getLogger("DELTA-RAG")


_STATUS_LABEL = {
    STATUS_HOLDS: "HOLDS",
    STATUS_SUPERSEDED: "superseded",
    STATUS_NOT_YET: "not yet in effect",
    STATUS_ALTERNATIVE: "competing",
    STATUS_UNDATED: "undated",
    STATUS_LATEST: "latest known",
}


@dataclass
class ACRConfig:

    render_unconflicted_slots: bool = True
    max_slots: int = 8
    max_assertions_per_slot: int = 6
    include_unslotted_events: bool = True
    max_unslotted_events: int = 40
    include_reading_directive: bool = True
    retain_alternatives: bool = True
    show_arbitration: bool = True


class ArbitratedContextRenderer:

    def __init__(self, config: Optional[ACRConfig] = None):
        self.config = config or ACRConfig()

    def render(
        self,
        results: Sequence[ArbitrationResult],
        unslotted_events: Sequence[Dict],
        anchor: Optional[str],
    ) -> str:
        cfg = self.config
        parts: List[str] = []

        renderable = [
            r for r in results
            if r.timeline.assertions
            and (cfg.render_unconflicted_slots or r.timeline.has_conflict)
        ]
        renderable.sort(
            key=lambda r: (len(r.timeline.conflicts), r.timeline.n_distinct_values),
            reverse=True,
        )
        renderable = renderable[: cfg.max_slots]

        if renderable:
            header = (
                f"## Resolved state timeline (query date: {anchor})"
                if anchor
                else "## Resolved state timeline (no explicit query date)"
            )
            parts.append(header)
            if cfg.include_reading_directive:
                if cfg.show_arbitration:
                    parts.append(
                        "Each block below is one subject-relation slot. Windows marked "
                        "HOLDS are the values in force at the query date; values marked "
                        "superseded had already ended, and values marked 'not yet in "
                        "effect' had not yet begun. A '~' marks a boundary inferred by "
                        "the system rather than stated in the source. Prefer the HOLDS "
                        "value unless the question asks about a different date."
                    )
                else:
                    parts.append(
                        "Each block below is one subject-relation slot, listing the "
                        "values recorded for it with the period over which each was "
                        "valid. A '~' marks a boundary inferred by the system rather "
                        "than stated in the source."
                    )
            parts.append("")

            for r in renderable:
                parts.append(self._render_slot(r))
                parts.append("")

        if cfg.include_unslotted_events and unslotted_events:
            parts.append("## Supporting events (chronological)")
            parts.append(self._render_events(unslotted_events))

        return "\n".join(parts).strip()


    def _render_slot(self, result: ArbitrationResult) -> str:
        cfg = self.config
        tl = result.timeline
        lines = [f"### {tl.slot_label}"]

        assertions = self._order_for_display(tl.assertions)

        if not cfg.retain_alternatives and result.decided and result.holder is not None:
            assertions = [result.holder]

        assertions = assertions[: cfg.max_assertions_per_slot]

        for a in assertions:
            lines.append(self._render_assertion(a))

        if not cfg.show_arbitration:
            return "\n".join(lines)

        if result.decided and result.holder is not None and self._assertable(result):
            lines.append(
                f"  → At {result.anchor}, {tl.subject} {self._relation_phrase(tl.relation)}: "
                f"{result.holder.value}"
            )
        elif tl.has_conflict:
            lines.append(
                "  → No value could be arbitrated at the query date; the windows "
                "above conflict and none clearly contains it."
            )

        return "\n".join(lines)

    @staticmethod
    def _assertable(result: ArbitrationResult) -> bool:
        holder = result.holder
        if holder is None:
            return False
        anchor_days = to_days(result.anchor)
        if anchor_days is None:
            return False
        return holder.interval.contains(anchor_days)

    @staticmethod
    def _order_for_display(assertions: Sequence[Assertion]) -> List[Assertion]:
        holders = [a for a in assertions if a.status == STATUS_HOLDS]
        rest = [a for a in assertions if a.status != STATUS_HOLDS]
        rest.sort(key=lambda a: (a.interval.start or "9999"))
        return holders + rest

    def _render_assertion(self, a: Assertion) -> str:
        corr = (
            f"  ({a.corroboration} sources)"
            if a.corroboration > 1
            else "  (1 source)"
        )
        if not self.config.show_arbitration:
            return f"    {a.value:<38} {a.interval.render()}{corr}"

        label = _STATUS_LABEL.get(a.status, a.status)
        marker = "▶" if a.status == STATUS_HOLDS else " "
        return (
            f"  {marker} {label:<18} {a.value:<38} "
            f"{a.interval.render()}{corr}"
        )

    @staticmethod
    def _relation_phrase(relation: str) -> str:
        return {
            "employer": "was employed by",
            "position": "held the position",
            "team": "played for",
            "educated_at": "was educated at",
            "spouse": "was married to",
            "residence": "resided in",
            "head_of": "was head of",
            "owner": "was owned by",
            "member_of": "was a member of",
            "award": "was awarded",
        }.get(relation, f"had {relation}")


    def _render_events(self, events: Sequence[Dict]) -> str:
        static_events, dated = [], []
        for ev in events:
            ts = ev.get("timestamp", "static")
            if not ts or ts in ("static", "unknown"):
                static_events.append(ev)
            else:
                dated.append(ev)
        dated.sort(key=lambda e: str(e.get("timestamp", "0")))

        ordered = (static_events + dated)[: self.config.max_unslotted_events]
        lines = []
        for i, ev in enumerate(ordered):
            ts = ev.get("timestamp", "static")
            display = "Static" if (not ts or ts in ("static", "unknown")) else ts
            lines.append(f"Event #{i + 1} [{display}]: {ev.get('sentence', '')}")
        return "\n".join(lines)
