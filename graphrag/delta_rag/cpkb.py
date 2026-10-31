import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

from .temporal_index import day_index

from .datp import TemporalKnowledgePartition

logger = logging.getLogger("DELTA-RAG")


@dataclass
class CPKBConfig:
    similarity_threshold: float = 0.6
    top_k_bridges_per_pair: int = 3
    max_total_bridges: int = 100
    temporal_decay_rate: float = 0.005
    entity_boost: float = 1.5
    min_partitions_for_bridging: int = 2


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
    return day_index(ts)


class CrossPartitionKGBridging:

    def __init__(self, config: CPKBConfig):
        self.config = config
        self.bridges: List[Dict] = []

    def compute_bridges(
        self,
        tkps: List[TemporalKnowledgePartition],
        events: List[Dict],
        embeddings: np.ndarray,
    ) -> List[Dict]:
        if len(tkps) < self.config.min_partitions_for_bridging:
            logger.info(
                f"CPKB: Only {len(tkps)} partitions, below minimum "
                f"{self.config.min_partitions_for_bridging}. No bridging needed."
            )
            return []

        partition_entities = self._build_partition_entity_sets(tkps, events)
        all_candidates = []

        for i in range(len(tkps)):
            for j in range(i + 1, len(tkps)):
                tkp_a = tkps[i]
                tkp_b = tkps[j]

                shared_entities = (
                    partition_entities[tkp_a.partition_id]
                    & partition_entities[tkp_b.partition_id]
                )
                entity_boost_val = (
                    self.config.entity_boost if shared_entities else 1.0
                )

                emb_a = embeddings[tkp_a.event_indices]
                emb_b = embeddings[tkp_b.event_indices]
                sim_matrix = _cosine_similarity_batch(emb_a, emb_b)

                for li in range(len(tkp_a.event_indices)):
                    for lj in range(len(tkp_b.event_indices)):
                        sim = sim_matrix[li, lj]
                        if sim < self.config.similarity_threshold:
                            continue

                        global_idx_a = tkp_a.event_indices[li]
                        global_idx_b = tkp_b.event_indices[lj]

                        ts_a = events[global_idx_a].get("timestamp", "static")
                        ts_b = events[global_idx_b].get("timestamp", "static")
                        temporal_decay = _temporal_proximity_decay(
                            ts_a, ts_b, self.config.temporal_decay_rate
                        )

                        bridge_weight = float(sim) * temporal_decay * entity_boost_val

                        all_candidates.append({
                            "source_event_idx": global_idx_a,
                            "target_event_idx": global_idx_b,
                            "source_event_id": events[global_idx_a].get("event_id", ""),
                            "target_event_id": events[global_idx_b].get("event_id", ""),
                            "source_partition": tkp_a.partition_id,
                            "target_partition": tkp_b.partition_id,
                            "bridge_weight": bridge_weight,
                            "semantic_similarity": float(sim),
                            "temporal_decay": temporal_decay,
                            "shared_entities": list(shared_entities) if shared_entities else [],
                        })

        self.n_candidates = len(all_candidates)
        all_candidates.sort(key=lambda x: x["bridge_weight"], reverse=True)

        pair_counts: Dict[Tuple[int, int], int] = {}
        selected = []

        for candidate in all_candidates:
            pair = (candidate["source_partition"], candidate["target_partition"])
            count = pair_counts.get(pair, 0)
            if count >= self.config.top_k_bridges_per_pair:
                continue
            selected.append(candidate)
            pair_counts[pair] = count + 1
            if len(selected) >= self.config.max_total_bridges:
                break

        self.bridges = selected

        n_pairs = len(set(
            (b["source_partition"], b["target_partition"]) for b in selected
        ))
        logger.info(
            f"CPKB: Created {len(selected)} bridges across {n_pairs} partition pairs"
        )
        if selected:
            avg_weight = np.mean([b["bridge_weight"] for b in selected])
            logger.info(f"CPKB: Average bridge weight: {avg_weight:.4f}")

        return self.bridges

    def _build_partition_entity_sets(
        self, tkps: List[TemporalKnowledgePartition], events: List[Dict]
    ) -> Dict[int, Set[str]]:
        partition_entities: Dict[int, Set[str]] = {}
        for tkp in tkps:
            entities = set()
            for idx in tkp.event_indices:
                for ent in events[idx].get("entities_involved", []):
                    entities.add(ent.lower())
            partition_entities[tkp.partition_id] = entities
        return partition_entities

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

    async def add_bridges_to_graph(self, event_graph, events: List[Dict]) -> int:
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
                    f"Cross-partition bridge between TKP-{bridge['source_partition']} "
                    f"and TKP-{bridge['target_partition']}"
                ),
                "is_undirected": True,
            }

            try:
                await event_graph.upsert_edge(src_id, tgt_id, edge_data)
                await event_graph.upsert_edge(tgt_id, src_id, edge_data)
                added += 1
            except Exception as e:
                logger.debug(f"CPKB: Could not add bridge {src_id}->{tgt_id}: {e}")

        logger.info(f"CPKB: Added {added} bridge edges to graph")
        return added

    def get_statistics(self) -> Dict:
        if not self.bridges:
            return {"total_bridges": 0}

        weights = [b["bridge_weight"] for b in self.bridges]
        partition_pairs = set(
            (b["source_partition"], b["target_partition"]) for b in self.bridges
        )
        bridges_with_shared_ents = sum(
            1 for b in self.bridges if b.get("shared_entities")
        )

        return {
            "total_bridges": len(self.bridges),
            "partition_pairs_bridged": len(partition_pairs),
            "avg_bridge_weight": float(np.mean(weights)),
            "min_bridge_weight": float(np.min(weights)),
            "max_bridge_weight": float(np.max(weights)),
            "bridges_with_shared_entities": bridges_with_shared_ents,
            "bridges_without_shared_entities": len(self.bridges) - bridges_with_shared_ents,
        }
