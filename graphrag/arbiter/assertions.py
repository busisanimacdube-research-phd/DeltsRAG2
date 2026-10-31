from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .temporal import (
    Interval,
    normalise_timestamp,
    parse_date_expression,
    parse_interval_expression,
    to_days,
)

logger = logging.getLogger("DELTA-RAG")


@dataclass(frozen=True)
class RelationType:
    name: str
    functional: bool
    triggers: Tuple[str, ...]


RELATION_LEXICON: Tuple[RelationType, ...] = (
    RelationType(
        "employer", True,
        (r"work(?:ed|s|ing)?\s+(?:for|at|with)", r"employed\s+(?:by|at)",
         r"join(?:ed|s)?", r"employer\s+(?:was|is)", r"hired\s+by",
         r"was\s+(?:a\s+)?(?:staff|employee)\s+(?:at|of)"),
    ),
    RelationType(
        "position", True,
        (r"serv(?:ed|es|ing)\s+as", r"was\s+appointed(?:\s+as)?",
         r"was\s+named(?:\s+as)?", r"became(?:\s+the)?",
         r"held\s+the\s+(?:position|post|office|role)\s+of",
         r"was\s+promoted\s+to", r"took\s+(?:up\s+)?the\s+(?:role|post)\s+of",
         r"was\s+elected(?:\s+as)?", r"assumed\s+the\s+(?:role|office|post)\s+of",
         r"was\s+(?:the\s+)?(?:chief executive|ceo|president|director|chairman|chair|"
         r"prime minister|chancellor|governor|secretary|head|manager|editor)\s+of"),
    ),
    RelationType(
        "team", True,
        (r"play(?:ed|s|ing)?\s+for", r"sign(?:ed|s)?\s+(?:for|with)",
         r"transferr?ed\s+to", r"was\s+(?:a\s+)?member\s+of\s+the\s+.{0,20}team"),
    ),
    RelationType(
        "educated_at", False,
        (r"stud(?:ied|ies)\s+at", r"graduat(?:ed|es)\s+from",
         r"enrol(?:led|ls)?\s+(?:at|in)", r"attend(?:ed|s)\s+",
         r"earned\s+.{0,30}\s+from", r"received\s+.{0,30}\s+degree\s+from"),
    ),
    RelationType(
        "spouse", True,
        (r"married(?:\s+to)?", r"wed(?:ded)?", r"spouse\s+(?:was|is)",
         r"was\s+the\s+(?:husband|wife)\s+of"),
    ),
    RelationType(
        "residence", True,
        (r"liv(?:ed|es|ing)\s+in", r"resid(?:ed|es|ing)\s+in",
         r"mov(?:ed|es)\s+to", r"relocat(?:ed|es)\s+to",
         r"was\s+based\s+in", r"settled\s+in"),
    ),
    RelationType(
        "head_of", True,
        (r"led\s+", r"head(?:ed|s)\s+", r"chair(?:ed|s)\s+",
         r"was\s+in\s+charge\s+of", r"presid(?:ed|es)\s+over",
         r"was\s+at\s+the\s+helm\s+of"),
    ),
    RelationType(
        "owner", True,
        (r"(?:was\s+)?(?:owned|acquired|purchased|bought)\s+by",
         r"was\s+a\s+subsidiary\s+of", r"belonged\s+to"),
    ),
    RelationType(
        "member_of", False,
        (r"was\s+a\s+member\s+of", r"belong(?:ed|s)\s+to",
         r"was\s+affiliated\s+with", r"was\s+part\s+of"),
    ),
    RelationType(
        "award", False,
        (r"(?:was\s+)?award(?:ed|s)?", r"(?:was\s+)?(?:won|wins|win)",
         r"received\s+the", r"was\s+honou?red\s+with"),
    ),
)

_COMPILED_TRIGGERS = [
    (rt, re.compile(r"\b(?:" + "|".join(rt.triggers) + r")\b", re.IGNORECASE))
    for rt in RELATION_LEXICON
]

_TERMINATOR_TRIGGERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("employer", (
        r"left", r"departed(?:\s+from)?", r"resigned\s+from", r"retired\s+from",
        r"quit", r"was\s+dismissed\s+from", r"was\s+fired\s+from",
        r"exited", r"parted\s+ways\s+with",
    )),
    ("position", (
        r"resigned\s+as", r"stepped\s+down\s+as", r"relinquished\s+the\s+(?:role|post|position)\s+of",
        r"vacated\s+the\s+(?:role|post|position|office)\s+of",
        r"stood\s+down\s+as", r"was\s+replaced\s+as", r"was\s+removed\s+as",
        r"ceased\s+to\s+be", r"ended\s+(?:his|her|their)\s+(?:term|tenure)\s+as",
    )),
    ("team", (r"left", r"was\s+released\s+by", r"was\s+transferred\s+from")),
    ("residence", (r"left", r"moved\s+away\s+from", r"moved\s+out\s+of")),
)

_COMPILED_TERMINATORS = [
    (rel, re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.IGNORECASE))
    for rel, pats in _TERMINATOR_TRIGGERS
]

_FUNCTIONAL_BY_NAME = {rt.name: rt.functional for rt in RELATION_LEXICON}


_TERMINATOR_RE = re.compile(
    r"\b(?:left|resigned|stepped\s+down|retired|departed|quit|"
    r"was\s+(?:succeeded|replaced|dismissed|fired|ousted|removed)|"
    r"ended|concluded|terminated|ceased|vacated|relinquished|"
    r"stood\s+down|handed\s+over|exited)\b",
    re.IGNORECASE,
)

_INITIATOR_RE = re.compile(
    r"\b(?:joined|began|started|became|was\s+appointed|was\s+named|"
    r"took\s+office|assumed|was\s+elected|signed|moved\s+to|founded|"
    r"was\s+hired|entered|commenced|was\s+promoted)\b",
    re.IGNORECASE,
)

_LEADING_NOISE = re.compile(
    r"^(?:the|a|an|as|to|of|at|in|for|with|by|its|his|her|their|new|then)\s+",
    re.IGNORECASE,
)

_TRAILING_CUT = re.compile(
    r"\s+(?:from|since|until|till|between|during|in|on|after|before|when|where|"
    r"which|who|while|and|but|because|although|following)\b.*$",
    re.IGNORECASE,
)

_PAREN_RE = re.compile(r"\([^)]*\)")


@dataclass
class Assertion:

    subject: str
    relation: str
    value: str
    interval: Interval
    functional: bool

    event_id: str = ""
    source_id: str = ""
    sentence: str = ""
    raw_timestamp: Optional[str] = None

    aspect: str = "occurrence"

    corroboration: int = 1
    superseded_by: Optional[str] = None
    status: str = "unresolved"
    closure: Optional[str] = None

    @property
    def slot(self) -> Tuple[str, str]:
        return (self.subject.lower().strip(), self.relation)

    @property
    def value_key(self) -> str:
        key = re.sub(r"[^a-z0-9 ]", " ", self.value.lower())
        return re.sub(r"\s+", " ", key).strip()

    def __repr__(self) -> str:
        return (
            f"Assertion({self.subject!r}, {self.relation}, {self.value!r}, "
            f"{self.interval.render()}, aspect={self.aspect})"
        )


@dataclass
class TANConfig:

    max_value_chars: int = 60
    min_value_chars: int = 2
    require_proper_noun: bool = True
    use_context_for_intervals: bool = True
    enable_interval_recovery: bool = True


def _normalise_entity(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _entity_matches(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and (a in b or b in a):
        return True
    aw, bw = set(a.split()), set(b.split())
    if not aw or not bw:
        return False
    inter = len(aw & bw)
    return bool(inter and inter / len(aw | bw) >= 0.6 and any(len(t) >= 4 for t in (aw & bw)))


def _clean_value(raw: str, cfg: TANConfig) -> Optional[str]:
    if not raw:
        return None
    v = _PAREN_RE.sub(" ", raw)
    v = v.split(";")[0].split(",")[0]
    v = _TRAILING_CUT.sub("", v)
    prev = None
    while prev != v:
        prev = v
        v = _LEADING_NOISE.sub("", v).strip()
    v = re.sub(r"[\s.]+$", "", v).strip()
    v = re.sub(r"\s{2,}", " ", v)
    v = re.sub(r"\b(1\d{3}|20\d{2})\b", "", v).strip(" -–—.,")

    if len(v) < cfg.min_value_chars or len(v) > cfg.max_value_chars:
        return None
    if cfg.require_proper_noun and not re.search(r"\b[A-Z][a-zA-Z]", v):
        return None
    if not re.search(r"[A-Za-z]", v):
        return None
    return v


def _detect_aspect(sentence: str) -> str:
    if _TERMINATOR_RE.search(sentence):
        return "terminate"
    if _INITIATOR_RE.search(sentence):
        return "initiate"
    return "occurrence"


def _recover_interval(
    sentence: str, context: str, raw_ts: Optional[str], aspect: str, cfg: TANConfig
) -> Interval:
    point = normalise_timestamp(raw_ts)

    if not cfg.enable_interval_recovery:
        if aspect == "terminate":
            return Interval(start=None, end=point, start_inferred=True)
        return Interval(start=point, end=None)

    expr = parse_interval_expression(sentence)
    if expr is None and cfg.use_context_for_intervals and context:
        expr = parse_interval_expression(context)

    if expr is not None:
        start, end = expr
        if start is None:
            start = point
        return Interval(
            start=start,
            end=end,
            start_inferred=(expr[0] is None and start is not None),
            end_inferred=False,
        )

    if aspect == "terminate":
        return Interval(start=None, end=point, start_inferred=True, end_inferred=False)
    return Interval(start=point, end=None, start_inferred=False, end_inferred=False)


class TypedAssertionNormaliser:

    def __init__(self, config: Optional[TANConfig] = None):
        self.config = config or TANConfig()

    def normalise(
        self,
        events: Sequence[Dict],
        query_entities: Sequence[str],
    ) -> Tuple[List[Assertion], List[Dict]]:
        pairs = [(e, _normalise_entity(e)) for e in (query_entities or [])]
        pairs = [(e, q) for e, q in pairs if q]
        canonical_entities = [e for e, _ in pairs]
        q_norms = [q for _, q in pairs]

        assertions: List[Assertion] = []
        unslotted: List[Dict] = []

        for ev in events:
            subject = self._resolve_subject(ev, canonical_entities, q_norms)
            if subject is None:
                unslotted.append(ev)
                continue

            typed = self._type_event(ev, subject)
            if typed is None:
                unslotted.append(ev)
                continue
            assertions.append(typed)

        logger.info(
            f"ARBITER/TAN: typed {len(assertions)} assertions from {len(events)} events "
            f"({len(unslotted)} unslotted)"
        )
        return assertions, unslotted

    def _resolve_subject(
        self, event: Dict, query_entities: Sequence[str], q_norms: Sequence[str]
    ) -> Optional[str]:
        if not q_norms:
            return None
        ents = event.get("entities_involved") or []
        if isinstance(ents, str):
            ents = [ents]
        for ent in ents:
            e_norm = _normalise_entity(ent)
            for canonical, q in zip(query_entities, q_norms):
                if _entity_matches(q, e_norm):
                    return canonical
        sent_norm = _normalise_entity(event.get("sentence", ""))
        for canonical, q in zip(query_entities, q_norms):
            if len(q) >= 4 and q in sent_norm:
                return canonical
        return None

    def _type_event(self, event: Dict, subject: str) -> Optional[Assertion]:
        sentence = str(event.get("sentence", "") or "")
        context = str(event.get("context", "") or "")
        if not sentence:
            return None

        aspect = _detect_aspect(sentence)

        matchers = (
            (self._match_terminator, self._match_relation)
            if aspect == "terminate"
            else (self._match_relation, self._match_terminator)
        )

        typed = None
        for matcher in matchers:
            typed = matcher(sentence, subject)
            if typed is not None:
                break
        if typed is None:
            return None

        relation, functional, value, via_terminator = typed
        if via_terminator:
            aspect = "terminate"

        interval = _recover_interval(
            sentence, context, event.get("timestamp"), aspect, self.config
        )
        return Assertion(
            subject=subject,
            relation=relation,
            value=value,
            interval=interval,
            functional=functional,
            event_id=str(event.get("id", "") or ""),
            source_id=str(event.get("source_id", "") or ""),
            sentence=sentence,
            raw_timestamp=event.get("timestamp"),
            aspect=aspect,
        )

    def _match_relation(
        self, sentence: str, subject: str
    ) -> Optional[Tuple[str, bool, str, bool]]:
        for rel, pattern in _COMPILED_TRIGGERS:
            m = pattern.search(sentence)
            if not m:
                continue
            value = self._value_after(sentence, m.end(), subject)
            if value:
                return (rel.name, rel.functional, value, False)
        return None

    def _match_terminator(
        self, sentence: str, subject: str
    ) -> Optional[Tuple[str, bool, str, bool]]:
        for rel_name, pattern in _COMPILED_TERMINATORS:
            m = pattern.search(sentence)
            if not m:
                continue
            value = self._value_after(sentence, m.end(), subject)
            if value:
                return (rel_name, _FUNCTIONAL_BY_NAME.get(rel_name, True), value, True)
        return None

    def _value_after(self, sentence: str, pos: int, subject: str) -> Optional[str]:
        tail = sentence[pos:]
        value = _clean_value(tail[: self.config.max_value_chars * 2], self.config)
        if not value:
            return None
        if _entity_matches(_normalise_entity(subject), _normalise_entity(value)):
            return None
        return value
