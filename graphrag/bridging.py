import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

from .clustering import TemporalEventCommunity

logger = logging.getLogger("CAST-RAG")


@dataclass
class BridgingConfig:
    similarity_threshold: float = 0.6
    top_k_bridges_per_cluster_pair: int = 3
    max_total_bridges: int = 100
    temporal_decay_rate: float = 0.005
    entity_priority_boost: float = 1.5
    min_clusters_for_bridging: int = 2


def _cosine_similarity_batch(
    embeddings_a: np.ndarray, embeddings_b: np.ndarray
) -> np.ndarray:
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    norms_a = np.where(norms_a == 0, 1, norms_a)
    norms_b = np.where(norms_b == 0, 1, norms_b)

    a_norm = embeddings_a / norms_a
    b_norm = embeddings_b / norms_b

    return a_norm @ b_norm.T


def _temporal_proximity_decay(
    ts1: Optional[str], ts2: Optional[str], decay_rate: float = 0.005
) -> float:
    if not ts1 or not ts2 or ts1 == "static" or ts2 == "static":
        return 0.5

    try:
        days1 = _timestamp_to_days(ts1)
        days2 = _timestamp_to_days(ts2)
        if days1 is None or days2 is None:
            return 0.5
        day_diff = abs(days1 - days2)
        return math.exp(-decay_rate * day_diff)
    except (ValueError, IndexError):
        return 0.5


def _timestamp_to_days(ts: str) -> Optional[int]:
    if not ts or ts == "static":
        return None
    try:
        parts = ts.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return year * 365 + month * 30 + day
    except (ValueError, IndexError):
        return None


class SemanticEventBridging:

    def __init__(self, config: BridgingConfig):
        self.config = config
        self.bridges: List[Dict] = []

    def compute_bridges(
        self,
        tecs: List[TemporalEventCommunity],
        events: List[Dict],
        embeddings: np.ndarray,
    ) -> List[Dict]:
        if len(tecs) < self.config.min_clusters_for_bridging:
            logger.info(
                f"SEB: Only {len(tecs)} clusters, below minimum "
                f"{self.config.min_clusters_for_bridging}. No bridging needed."
            )
            return []

        cluster_entities = self._build_cluster_entity_sets(tecs, events)

        all_candidates = []

        for i in range(len(tecs)):
            for j in range(i + 1, len(tecs)):
                tec_a = tecs[i]
                tec_b = tecs[j]

                shared_entities = cluster_entities[tec_a.cluster_id] & cluster_entities[
                    tec_b.cluster_id
                ]
                entity_boost = (
                    self.config.entity_priority_boost if shared_entities else 1.0
                )

                emb_a = embeddings[tec_a.event_indices]
                emb_b = embeddings[tec_b.event_indices]

                sim_matrix = _cosine_similarity_batch(emb_a, emb_b)

                for li in range(len(tec_a.event_indices)):
                    for lj in range(len(tec_b.event_indices)):
                        sim = sim_matrix[li, lj]
                        if sim < self.config.similarity_threshold:
                            continue

                        global_idx_a = tec_a.event_indices[li]
                        global_idx_b = tec_b.event_indices[lj]

                        ts_a = events[global_idx_a].get("timestamp", "static")
                        ts_b = events[global_idx_b].get("timestamp", "static")

                        temporal_decay = _temporal_proximity_decay(
                            ts_a, ts_b, self.config.temporal_decay_rate
                        )

                        bridge_weight = float(sim) * temporal_decay * entity_boost

                        all_candidates.append(
                            {
                                "source_event_idx": global_idx_a,
                                "target_event_idx": global_idx_b,
                                "source_event_id": events[global_idx_a].get(
                                    "event_id", ""
                                ),
                                "target_event_id": events[global_idx_b].get(
                                    "event_id", ""
                                ),
                                "source_cluster": tec_a.cluster_id,
                                "target_cluster": tec_b.cluster_id,
                                "bridge_weight": bridge_weight,
                                "semantic_similarity": float(sim),
                                "temporal_decay": temporal_decay,
                                "shared_entities": list(shared_entities) if shared_entities else [],
                            }
                        )

        all_candidates.sort(key=lambda x: x["bridge_weight"], reverse=True)

        pair_counts: Dict[Tuple[int, int], int] = {}
        selected = []

        for candidate in all_candidates:
            pair = (candidate["source_cluster"], candidate["target_cluster"])
            count = pair_counts.get(pair, 0)

            if count >= self.config.top_k_bridges_per_cluster_pair:
                continue

            selected.append(candidate)
            pair_counts[pair] = count + 1

            if len(selected) >= self.config.max_total_bridges:
                break

        self.bridges = selected

        n_pairs = len(set((b["source_cluster"], b["target_cluster"]) for b in selected))
        logger.info(
            f"SEB: Created {len(selected)} bridges across {n_pairs} cluster pairs"
        )
        if selected:
            avg_weight = np.mean([b["bridge_weight"] for b in selected])
            logger.info(f"SEB: Average bridge weight: {avg_weight:.4f}")

        return self.bridges

    def _build_cluster_entity_sets(
        self, tecs: List[TemporalEventCommunity], events: List[Dict]
    ) -> Dict[int, Set[str]]:
        cluster_entities: Dict[int, Set[str]] = {}

        for tec in tecs:
            entities = set()
            for idx in tec.event_indices:
                for ent in events[idx].get("entities_involved", []):
                    entities.add(ent.lower())
            cluster_entities[tec.cluster_id] = entities

        return cluster_entities

    def get_bridge_connectivity(self, event_idx: int) -> float:
        total = 0.0
        for bridge in self.bridges:
            if (
                bridge["source_event_idx"] == event_idx
                or bridge["target_event_idx"] == event_idx
            ):
                total += bridge["bridge_weight"]
        return total

    def get_bridged_events(self, event_idx: int) -> List[int]:
        connected = []
        for bridge in self.bridges:
            if bridge["source_event_idx"] == event_idx:
                connected.append(bridge["target_event_idx"])
            elif bridge["target_event_idx"] == event_idx:
                connected.append(bridge["source_event_idx"])
        return connected

    def add_bridges_to_graph(self, event_graph, events: List[Dict]):
        added = 0
        for bridge in self.bridges:
            src_id = bridge["source_event_id"]
            tgt_id = bridge["target_event_id"]

            if not src_id or not tgt_id:
                continue

            edge_data = {
                "relation_type": "semantic_bridge",
                "weight": bridge["bridge_weight"],
                "semantic_similarity": bridge["semantic_similarity"],
                "temporal_decay": bridge["temporal_decay"],
                "shared_entities": ",".join(bridge.get("shared_entities", [])),
                "description": (
                    f"Semantic bridge between clusters "
                    f"{bridge['source_cluster']} and {bridge['target_cluster']}"
                ),
                "is_undirected": True,
            }

            added += 1

        logger.info(f"SEB: Prepared {added} bridge edges for graph insertion")
        return added
