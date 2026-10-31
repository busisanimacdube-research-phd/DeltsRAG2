import math
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from .temporal_index import day_index

from .datp import TemporalKnowledgePartition
from .cpkb import CrossPartitionKGBridging

logger = logging.getLogger("DELTA-RAG")

TEMPORAL_KEYWORDS = {
    "when", "year", "date", "time", "before", "after", "during", "since",
    "until", "between", "from", "period", "era", "century", "decade",
    "month", "day", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "first", "last", "latest", "earliest", "recent", "previous", "next",
    "beginning", "end", "start", "finish", "prior", "following",
}

TEMPORAL_PATTERNS = [
    r"\b\d{4}\b",
    r"\b\d{4}-\d{2}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bin\s+\d{4}\b",
    r"\b(?:before|after|since|until)\s+\d{4}\b",
    r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
]


@dataclass
class AHKRConfig:
    top_p_partitions: int = 5
    partition_semantic_weight: float = 0.7
    partition_temporal_weight: float = 0.3

    alpha: float = 0.35
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.10
    epsilon: float = 0.20

    temporal_boost: float = 1.5
    entity_boost: float = 1.3

    non_selected_damping: float = 0.7

    top_k_events: int = 20
    normalize_scores: bool = True


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _temporal_alignment_score(
    query_time_constraints: Dict,
    event_timestamp: str,
) -> float:
    start_time = query_time_constraints.get("start_time")
    end_time = query_time_constraints.get("end_time")

    if not event_timestamp or event_timestamp == "static":
        return 0.3

    event_days = _ts_to_days(event_timestamp)
    if event_days is None:
        return 0.3

    start_days = _ts_to_days(start_time) if start_time else None
    end_days = _ts_to_days(end_time) if end_time else None

    if start_days is not None and end_days is not None:
        if start_days <= event_days <= end_days:
            return 1.0
        else:
            dist = min(abs(event_days - start_days), abs(event_days - end_days))
            return max(0.0, math.exp(-0.002 * dist))
    elif start_days is not None:
        dist = abs(event_days - start_days)
        return max(0.0, math.exp(-0.002 * dist))
    elif end_days is not None:
        dist = abs(event_days - end_days)
        return max(0.0, math.exp(-0.002 * dist))

    return 0.5


def _ts_to_days(ts: Optional[str]) -> Optional[int]:
    return day_index(ts)


def _partition_temporal_alignment(
    tkp: TemporalKnowledgePartition, query_time_constraints: Dict
) -> float:
    earliest, latest = tkp.temporal_span
    if earliest is None and latest is None:
        return 0.3

    scores = []
    if earliest:
        scores.append(_temporal_alignment_score(query_time_constraints, earliest))
    if latest:
        scores.append(_temporal_alignment_score(query_time_constraints, latest))

    return max(scores) if scores else 0.3


def _entity_match_score(
    query_entities: List[str],
    event_entities: List[str],
) -> float:
    if not query_entities or not event_entities:
        return 0.0

    query_lower = [e.lower().strip() for e in query_entities if e]
    event_lower = [e.lower().strip() for e in event_entities if e]

    if not query_lower or not event_lower:
        return 0.0

    matches = 0.0
    for qe in query_lower:
        best = 0.0
        qe_words = set(qe.split())
        for ee in event_lower:
            if qe == ee:
                best = 1.0
                break
            if qe in ee or ee in qe:
                best = max(best, 0.8)
                continue
            ee_words = set(ee.split())
            if qe_words and ee_words:
                overlap = len(qe_words & ee_words) / len(qe_words | ee_words)
                if overlap > 0:
                    best = max(best, 0.5 * overlap)
        matches += best

    return matches / len(query_lower)


class AdaptiveHierarchicalRetrieval:

    def __init__(self, config: AHKRConfig):
        self.config = config
        self._last_query_type = None
        self._last_adapted_weights = None

    def retrieve(
        self,
        query_embedding: np.ndarray,
        tkps: List[TemporalKnowledgePartition],
        events: List[Dict],
        embeddings: np.ndarray,
        time_constraints: Dict,
        cpkb: Optional[CrossPartitionKGBridging] = None,
        entities: Optional[List[str]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict]:
        if not tkps:
            logger.warning("AHKR: No TKPs available for retrieval")
            return []

        query_info = self._classify_query(
            query_text or "", time_constraints, entities or []
        )
        self._last_query_type = query_info["label"]

        alpha, beta, gamma, delta, epsilon = self._adapt_weights(query_info)
        self._last_adapted_weights = (alpha, beta, gamma, delta, epsilon)

        logger.info(
            f"AHKR: Query type={query_info['label']} "
            f"(temporal={query_info['temporal_strength']:.2f}, "
            f"entity={query_info['entity_strength']:.2f}), "
            f"weights=(α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, "
            f"δ={delta:.2f}, ε={epsilon:.2f})"
        )

        selected_partitions = self._phase1_partition_selection(
            query_embedding, tkps, embeddings, time_constraints
        )

        if not selected_partitions:
            logger.warning("AHKR: No partitions selected in Phase 1")
            return []

        selected_ids = {tkp.partition_id for tkp, _ in selected_partitions}

        logger.info(
            f"AHKR Phase 1: Selected {len(selected_partitions)} partitions "
            f"from {len(tkps)} total"
        )

        all_partitions_with_scores = []
        for tkp, score in selected_partitions:
            all_partitions_with_scores.append((tkp, score, True))
        for tkp in tkps:
            if tkp.partition_id not in selected_ids:
                all_partitions_with_scores.append(
                    (tkp, 0.0, False)
                )

        retrieved_events = self._phase2_event_retrieval(
            query_embedding,
            all_partitions_with_scores,
            events,
            embeddings,
            time_constraints,
            cpkb,
            entities,
            alpha, beta, gamma, delta, epsilon,
        )

        logger.info(f"AHKR Phase 2: Retrieved {len(retrieved_events)} events")
        return retrieved_events

    def _classify_query(
        self,
        query_text: str,
        time_constraints: Dict,
        entities: List[str],
    ) -> Dict[str, float]:
        query_lower = query_text.lower()

        has_time_constraints = bool(
            time_constraints.get("start_time") or time_constraints.get("end_time")
        )
        temporal_keyword_count = sum(
            1 for kw in TEMPORAL_KEYWORDS if kw in query_lower.split()
        )
        temporal_pattern_count = sum(
            1 for pat in TEMPORAL_PATTERNS if re.search(pat, query_text)
        )
        temporal_strength = 0.0
        if has_time_constraints:
            temporal_strength += 0.4
        temporal_strength += min(temporal_keyword_count * 0.15, 0.3)
        temporal_strength += min(temporal_pattern_count * 0.2, 0.3)
        temporal_strength = min(temporal_strength, 1.0)

        entity_count = len(entities) if entities else 0
        entity_strength = min(entity_count * 0.3, 1.0)

        if temporal_strength > 0.5 and entity_strength <= 0.3:
            label = "temporal_dominant"
        elif entity_strength > 0.5 and temporal_strength <= 0.3:
            label = "entity_dominant"
        else:
            label = "hybrid"

        return {
            "temporal_strength": temporal_strength,
            "entity_strength": entity_strength,
            "label": label,
        }

    def _adapt_weights(
        self, query_info: Dict[str, float]
    ) -> Tuple[float, float, float, float, float]:
        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma
        delta = self.config.delta
        epsilon = self.config.epsilon

        temporal_strength = query_info["temporal_strength"]
        entity_strength = query_info["entity_strength"]

        temporal_factor = 1.0 + (self.config.temporal_boost - 1.0) * temporal_strength
        beta *= temporal_factor
        gamma *= temporal_factor

        entity_factor = 1.0 + (self.config.entity_boost - 1.0) * entity_strength
        alpha *= entity_factor
        epsilon *= entity_factor

        total = alpha + beta + gamma + delta + epsilon
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
            delta /= total
            epsilon /= total

        return alpha, beta, gamma, delta, epsilon

    def _phase1_partition_selection(
        self,
        query_embedding: np.ndarray,
        tkps: List[TemporalKnowledgePartition],
        embeddings: np.ndarray,
        time_constraints: Dict,
    ) -> List[Tuple[TemporalKnowledgePartition, float]]:
        partition_scores = []

        for tkp in tkps:
            semantic_score = _cosine_sim(query_embedding, tkp.centroid)
            temporal_score = _partition_temporal_alignment(tkp, time_constraints)

            combined = (
                self.config.partition_semantic_weight * semantic_score
                + self.config.partition_temporal_weight * temporal_score
            )
            partition_scores.append((tkp, combined))

        partition_scores.sort(key=lambda x: x[1], reverse=True)
        min_select = max(self.config.top_p_partitions, len(partition_scores) // 2)
        top_p = min(min_select, len(partition_scores))
        return partition_scores[:top_p]

    def _phase2_event_retrieval(
        self,
        query_embedding: np.ndarray,
        all_partitions: List[Tuple],
        events: List[Dict],
        embeddings: np.ndarray,
        time_constraints: Dict,
        cpkb: Optional[CrossPartitionKGBridging],
        entities: Optional[List[str]],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float,
    ) -> List[Dict]:
        scored_events = []
        damping = self.config.non_selected_damping

        bridge_scores = {}
        if cpkb and cpkb.bridges:
            max_bridge = max(
                (b["bridge_weight"] for b in cpkb.bridges), default=1.0
            )
            for bridge in cpkb.bridges:
                src = bridge["source_event_idx"]
                tgt = bridge["target_event_idx"]
                normalized = bridge["bridge_weight"] / max(max_bridge, 1e-8)
                bridge_scores[src] = bridge_scores.get(src, 0) + normalized
                bridge_scores[tgt] = bridge_scores.get(tgt, 0) + normalized

            if bridge_scores:
                max_bs = max(bridge_scores.values())
                if max_bs > 0:
                    bridge_scores = {k: v / max_bs for k, v in bridge_scores.items()}

        for entry in all_partitions:
            tkp, partition_score, is_selected = entry
            for event_idx in tkp.event_indices:
                if event_idx >= len(events) or event_idx >= len(embeddings):
                    continue

                event = events[event_idx]

                semantic_sim = _cosine_sim(query_embedding, embeddings[event_idx])
                temporal_sim = _temporal_alignment_score(
                    time_constraints, event.get("timestamp", "static")
                )
                partition_conf = tkp.confidence_scores.get(event_idx, 0.5)
                bridge_conn = bridge_scores.get(event_idx, 0.0)
                entity_match = _entity_match_score(
                    entities or [], event.get("entities_involved", [])
                )

                combined_score = (
                    alpha * semantic_sim
                    + beta * temporal_sim
                    + gamma * partition_conf
                    + delta * bridge_conn
                    + epsilon * entity_match
                )

                if not is_selected:
                    combined_score *= damping

                scored_events.append({
                    "event_idx": event_idx,
                    "id": event.get("event_id", ""),
                    "timestamp": event.get("timestamp", "static"),
                    "sentence": event.get("sentence", ""),
                    "context": event.get("context", ""),
                    "entities_involved": event.get("entities_involved", []),
                    "source_id": event.get("source_id", ""),
                    "combined_score": combined_score,
                    "semantic_sim": semantic_sim,
                    "temporal_sim": temporal_sim,
                    "partition_confidence": partition_conf,
                    "bridge_connectivity": bridge_conn,
                    "entity_match": entity_match,
                    "partition_id": tkp.partition_id,
                    "query_type": self._last_query_type,
                    "distance": 1.0 - combined_score,
                })

        best_by_idx = {}
        for ev in scored_events:
            idx = ev["event_idx"]
            if idx not in best_by_idx or ev["combined_score"] > best_by_idx[idx]["combined_score"]:
                best_by_idx[idx] = ev
        unique_events = list(best_by_idx.values())

        unique_events.sort(key=lambda x: x["combined_score"], reverse=True)

        if self.config.normalize_scores and unique_events:
            max_score = unique_events[0]["combined_score"]
            min_score = unique_events[-1]["combined_score"]
            score_range = max_score - min_score
            if score_range > 0:
                for ev in unique_events:
                    ev["combined_score"] = (
                        ev["combined_score"] - min_score
                    ) / score_range

        return unique_events[: self.config.top_k_events]

    @property
    def last_query_type(self) -> Optional[str]:
        return self._last_query_type

    @property
    def last_adapted_weights(self) -> Optional[Tuple[float, float, float, float, float]]:
        return self._last_adapted_weights
