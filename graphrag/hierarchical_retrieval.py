import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from .clustering import TemporalEventCommunity
from .bridging import SemanticEventBridging

logger = logging.getLogger("CAST-RAG")


@dataclass
class HierarchicalRetrievalConfig:
    top_c_clusters: int = 5
    cluster_temporal_weight: float = 0.3
    cluster_semantic_weight: float = 0.7

    alpha: float = 0.4
    beta: float = 0.25
    gamma: float = 0.2
    delta: float = 0.15

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
    if not ts or ts == "static" or ts == "0":
        return None
    try:
        parts = ts.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return year * 365 + month * 30 + day
    except (ValueError, IndexError):
        return None


def _cluster_temporal_alignment(
    tec: TemporalEventCommunity, query_time_constraints: Dict
) -> float:
    earliest, latest = tec.temporal_span

    if earliest is None and latest is None:
        return 0.3

    scores = []
    if earliest:
        scores.append(
            _temporal_alignment_score(query_time_constraints, earliest)
        )
    if latest:
        scores.append(
            _temporal_alignment_score(query_time_constraints, latest)
        )

    return max(scores) if scores else 0.3


class ConfidenceWeightedHierarchicalRetrieval:

    def __init__(self, config: HierarchicalRetrievalConfig):
        self.config = config

    def retrieve(
        self,
        query_embedding: np.ndarray,
        tecs: List[TemporalEventCommunity],
        events: List[Dict],
        embeddings: np.ndarray,
        time_constraints: Dict,
        bridging: Optional[SemanticEventBridging] = None,
        entities: Optional[List[str]] = None,
    ) -> List[Dict]:
        if not tecs:
            logger.warning("CWHR: No TECs available for retrieval")
            return []

        selected_clusters = self._phase1_cluster_selection(
            query_embedding, tecs, embeddings, time_constraints
        )

        if not selected_clusters:
            logger.warning("CWHR: No clusters selected in Phase 1")
            return []

        logger.info(
            f"CWHR Phase 1: Selected {len(selected_clusters)} clusters "
            f"from {len(tecs)} total"
        )

        retrieved_events = self._phase2_event_retrieval(
            query_embedding,
            selected_clusters,
            events,
            embeddings,
            time_constraints,
            bridging,
            entities,
        )

        logger.info(
            f"CWHR Phase 2: Retrieved {len(retrieved_events)} events"
        )

        return retrieved_events

    def _phase1_cluster_selection(
        self,
        query_embedding: np.ndarray,
        tecs: List[TemporalEventCommunity],
        embeddings: np.ndarray,
        time_constraints: Dict,
    ) -> List[Tuple[TemporalEventCommunity, float]]:
        cluster_scores = []

        for tec in tecs:
            semantic_score = _cosine_sim(query_embedding, tec.centroid)

            temporal_score = _cluster_temporal_alignment(tec, time_constraints)

            combined = (
                self.config.cluster_semantic_weight * semantic_score
                + self.config.cluster_temporal_weight * temporal_score
            )

            cluster_scores.append((tec, combined))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        top_c = min(self.config.top_c_clusters, len(cluster_scores))

        return cluster_scores[:top_c]

    def _phase2_event_retrieval(
        self,
        query_embedding: np.ndarray,
        selected_clusters: List[Tuple[TemporalEventCommunity, float]],
        events: List[Dict],
        embeddings: np.ndarray,
        time_constraints: Dict,
        bridging: Optional[SemanticEventBridging],
        entities: Optional[List[str]],
    ) -> List[Dict]:
        scored_events = []

        bridge_scores = {}
        if bridging and bridging.bridges:
            max_bridge = max(
                (b["bridge_weight"] for b in bridging.bridges), default=1.0
            )
            for bridge in bridging.bridges:
                src = bridge["source_event_idx"]
                tgt = bridge["target_event_idx"]
                normalized = bridge["bridge_weight"] / max(max_bridge, 1e-8)
                bridge_scores[src] = bridge_scores.get(src, 0) + normalized
                bridge_scores[tgt] = bridge_scores.get(tgt, 0) + normalized

        if bridge_scores:
            max_bs = max(bridge_scores.values())
            if max_bs > 0:
                bridge_scores = {k: v / max_bs for k, v in bridge_scores.items()}

        for tec, cluster_score in selected_clusters:
            for event_idx in tec.event_indices:
                if event_idx >= len(events) or event_idx >= len(embeddings):
                    continue

                event = events[event_idx]

                semantic_sim = _cosine_sim(
                    query_embedding, embeddings[event_idx]
                )

                temporal_sim = _temporal_alignment_score(
                    time_constraints, event.get("timestamp", "static")
                )

                cluster_conf = tec.confidence_scores.get(event_idx, 0.5)

                bridge_conn = bridge_scores.get(event_idx, 0.0)

                combined_score = (
                    self.config.alpha * semantic_sim
                    + self.config.beta * temporal_sim
                    + self.config.gamma * cluster_conf
                    + self.config.delta * bridge_conn
                )

                scored_events.append(
                    {
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
                        "cluster_confidence": cluster_conf,
                        "bridge_connectivity": bridge_conn,
                        "cluster_id": tec.cluster_id,
                        "distance": 1.0 - combined_score,
                    }
                )

        seen = set()
        unique_events = []
        for ev in scored_events:
            if ev["event_idx"] not in seen:
                seen.add(ev["event_idx"])
                unique_events.append(ev)

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
