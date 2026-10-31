import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from .temporal_index import day_index

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger("DELTA-RAG")


@dataclass
class DATPConfig:
    min_cluster_size_range: Tuple[int, ...] = (3, 5, 7, 10)
    min_samples: int = 2
    cluster_selection_method: str = "eom"
    metric: str = "euclidean"
    fourier_dim: int = 16
    noise_reassignment: bool = True
    noise_confidence_decay: float = 0.5
    auto_tune: bool = True
    temporal_weight: float = 1.0
    span_from_core_only: bool = True


@dataclass
class TemporalKnowledgePartition:
    partition_id: int
    event_indices: List[int]
    event_ids: List[str]
    centroid: np.ndarray
    confidence_scores: Dict[int, float]
    temporal_span: Tuple[Optional[str], Optional[str]]
    avg_confidence: float = 0.0

    def __post_init__(self):
        if self.confidence_scores:
            self.avg_confidence = float(np.mean(list(self.confidence_scores.values())))


def fourier_timestamp_encoding(timestamps: List[str], dim: int = 16) -> np.ndarray:
    assert dim % 2 == 0, "dim must be even"
    n_f = dim // 2
    days = [day_index(ts) for ts in timestamps]
    dated = [d for d in days if d is not None]
    features = np.zeros((len(timestamps), dim), dtype=np.float32)
    if not dated:
        return features
    lo, hi = min(dated), max(dated)
    span = max(hi - lo, 1)
    freqs = [2.0 ** (j - 2) for j in range(1, n_f + 1)]
    for i, d in enumerate(days):
        if d is None:
            continue
        t = (d - lo) / span
        for j, f in enumerate(freqs):
            features[i, 2 * j] = math.sin(2 * math.pi * f * t)
            features[i, 2 * j + 1] = math.cos(2 * math.pi * f * t)
    return features


def _parse_timestamp_components(
    ts: str,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not ts or ts == "static":
        return None, None, None
    try:
        parts = ts.split("-")
        year = int(parts[0]) if len(parts) > 0 else None
        month = int(parts[1]) if len(parts) > 1 else None
        day = int(parts[2]) if len(parts) > 2 else None
        return year, month, day
    except (ValueError, IndexError):
        return None, None, None


class DensityAdaptiveTemporalPartitioning:

    def __init__(self, config: DATPConfig):
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan package required for DATP. Install with: pip install hdbscan"
            )
        self.config = config
        self.labels_ = None
        self.probabilities_ = None
        self.tkps: List[TemporalKnowledgePartition] = []
        self._best_min_cluster_size = None
        self.tuning_scores_: Dict[int, float] = {}
        self.noise_mask_: Optional[np.ndarray] = None

    def fit(
        self, events: List[Dict], mstc_embeddings: np.ndarray
    ) -> List[TemporalKnowledgePartition]:
        n = len(events)
        if n < 3:
            logger.warning(f"DATP: Only {n} events, too few for clustering.")
            return self._single_partition_fallback(events, mstc_embeddings)

        timestamps = [e.get("timestamp", "static") for e in events]
        fourier_features = fourier_timestamp_encoding(
            timestamps, dim=self.config.fourier_dim
        )

        z = mstc_embeddings / np.maximum(
            np.linalg.norm(mstc_embeddings, axis=1, keepdims=True), 1e-12)
        n_f = self.config.fourier_dim // 2
        psi = fourier_features * (self.config.temporal_weight / math.sqrt(n_f))
        combined = np.concatenate([z, psi], axis=1).astype(np.float32)

        logger.info(
            f"DATP: Combined feature dim = {combined.shape[1]} "
            f"(MSTC: {mstc_embeddings.shape[1]}, Fourier: {fourier_features.shape[1]})"
        )

        if self.config.auto_tune and n >= 20:
            best_size, best_score = self._auto_tune(combined)
            logger.info(
                f"DATP: Auto-tuned min_cluster_size={best_size} "
                f"(validity={best_score:.4f})"
            )
            self._best_min_cluster_size = best_size
        else:
            self._best_min_cluster_size = min(
                self.config.min_cluster_size_range[0], max(3, n // 5)
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._best_min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_method=self.config.cluster_selection_method,
            metric=self.config.metric,
            prediction_data=True,
        )
        clusterer.fit(combined)

        self.labels_ = clusterer.labels_
        self.probabilities_ = clusterer.probabilities_

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        logger.info(f"DATP: Found {n_clusters} partitions, {n_noise} noise points")

        self.noise_mask_ = self.labels_ == -1

        if self.config.noise_reassignment and n_noise > 0 and n_clusters > 0:
            self._reassign_noise(combined)

        self.tkps = self._build_tkps(events, mstc_embeddings)
        logger.info(f"DATP: Created {len(self.tkps)} Temporal Knowledge Partitions")
        return self.tkps

    def _auto_tune(self, features: np.ndarray) -> Tuple[int, float]:
        best_size = self.config.min_cluster_size_range[0]
        best_score = -1.0

        for size in self.config.min_cluster_size_range:
            if size > len(features) // 2:
                continue
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=size,
                    min_samples=self.config.min_samples,
                    cluster_selection_method=self.config.cluster_selection_method,
                    metric=self.config.metric,
                    gen_min_span_tree=True,
                )
                clusterer.fit(features)

                labels = clusterer.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    continue

                score = float(clusterer.relative_validity_)
                self.tuning_scores_[size] = score

                if score > best_score:
                    best_score = score
                    best_size = size
            except Exception as e:
                logger.warning(f"DATP auto-tune: size={size} failed: {e}")
                continue

        return best_size, best_score

    def _reassign_noise(self, features: np.ndarray):
        noise_indices = np.where(self.labels_ == -1)[0]
        if len(noise_indices) == 0:
            return

        unique_labels = set(self.labels_)
        unique_labels.discard(-1)
        centroids = {}
        for label in unique_labels:
            mask = self.labels_ == label
            centroids[label] = features[mask].mean(axis=0)

        for idx in noise_indices:
            point = features[idx]
            min_dist = float("inf")
            best_label = 0
            for label, centroid in centroids.items():
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label

            self.labels_[idx] = best_label
            self.probabilities_[idx] *= self.config.noise_confidence_decay

        logger.info(f"DATP: Reassigned {len(noise_indices)} noise points")

    def _build_tkps(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> List[TemporalKnowledgePartition]:
        unique_labels = sorted(set(self.labels_))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        tkps = []
        for label in unique_labels:
            mask = self.labels_ == label
            indices = np.where(mask)[0].tolist()

            event_ids = []
            timestamps_in_partition = []
            confidence_scores = {}

            for idx in indices:
                event_ids.append(events[idx].get("event_id", f"event-{idx}"))
                ts = events[idx].get("timestamp", "static")
                is_noise = self.noise_mask_ is not None and bool(self.noise_mask_[idx])
                if ts and ts != "static" and not (
                        self.config.span_from_core_only and is_noise):
                    timestamps_in_partition.append(ts)
                confidence_scores[idx] = float(self.probabilities_[idx])

            centroid = embeddings[mask].mean(axis=0)
            timestamps_in_partition.sort(key=lambda t: day_index(t) or 0)
            earliest = timestamps_in_partition[0] if timestamps_in_partition else None
            latest = timestamps_in_partition[-1] if timestamps_in_partition else None

            tkp = TemporalKnowledgePartition(
                partition_id=label,
                event_indices=indices,
                event_ids=event_ids,
                centroid=centroid,
                confidence_scores=confidence_scores,
                temporal_span=(earliest, latest),
            )
            tkps.append(tkp)

        return tkps

    def _single_partition_fallback(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> List[TemporalKnowledgePartition]:
        indices = list(range(len(events)))
        event_ids = [e.get("event_id", f"event-{i}") for i, e in enumerate(events)]
        timestamps = [
            e.get("timestamp", "static")
            for e in events
            if e.get("timestamp", "static") != "static"
        ]

        self.labels_ = np.zeros(len(events), dtype=int)
        self.probabilities_ = np.ones(len(events), dtype=float)

        tkp = TemporalKnowledgePartition(
            partition_id=0,
            event_indices=indices,
            event_ids=event_ids,
            centroid=embeddings.mean(axis=0),
            confidence_scores={i: 1.0 for i in indices},
            temporal_span=(
                min(timestamps) if timestamps else None,
                max(timestamps) if timestamps else None,
            ),
        )
        self.tkps = [tkp]
        return self.tkps

    def get_event_partition(self, event_index: int) -> Optional[int]:
        if self.labels_ is not None and event_index < len(self.labels_):
            return int(self.labels_[event_index])
        return None

    def get_event_confidence(self, event_index: int) -> float:
        if self.probabilities_ is not None and event_index < len(self.probabilities_):
            return float(self.probabilities_[event_index])
        return 0.0

    def get_partition_by_id(
        self, partition_id: int
    ) -> Optional[TemporalKnowledgePartition]:
        for tkp in self.tkps:
            if tkp.partition_id == partition_id:
                return tkp
        return None
