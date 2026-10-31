import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger("CAST-RAG")


@dataclass
class ClusteringConfig:
    min_cluster_size_range: Tuple[int, ...] = (3, 5, 7, 10)
    min_samples: int = 2
    cluster_selection_method: str = "eom"
    metric: str = "euclidean"
    timestamp_fourier_dim: int = 16
    noise_reassignment: bool = True
    noise_confidence_decay: float = 0.5
    auto_tune: bool = True


@dataclass
class TemporalEventCommunity:

    cluster_id: int
    event_indices: List[int]
    event_ids: List[str]
    centroid: np.ndarray
    confidence_scores: Dict[int, float]
    temporal_span: Tuple[Optional[str], Optional[str]]
    avg_confidence: float = 0.0

    def __post_init__(self):
        if self.confidence_scores:
            self.avg_confidence = np.mean(list(self.confidence_scores.values()))


def fourier_timestamp_encoding(timestamps: List[str], dim: int = 16) -> np.ndarray:
    assert dim % 4 == 0, "dim must be divisible by 4"

    n = len(timestamps)
    features = np.zeros((n, dim), dtype=np.float32)
    n_freq = dim // 4

    for i, ts in enumerate(timestamps):
        year, month, day = _parse_timestamp_components(ts)

        year_norm = (year - 1900) / 200.0 if year else 0.0
        month_norm = month / 12.0 if month else 0.0
        day_norm = day / 31.0 if day else 0.0
        combined = year_norm * 365 + month_norm * 30 + day_norm

        for j in range(n_freq):
            freq = 2.0 ** j
            features[i, 4 * j] = math.sin(2 * math.pi * freq * year_norm)
            features[i, 4 * j + 1] = math.cos(2 * math.pi * freq * year_norm)
            features[i, 4 * j + 2] = math.sin(2 * math.pi * freq * combined)
            features[i, 4 * j + 3] = math.cos(2 * math.pi * freq * combined)

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


class AdaptiveTemporalClustering:

    def __init__(self, config: ClusteringConfig):
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan package required for ATEC. Install with: pip install hdbscan"
            )
        self.config = config
        self.labels_ = None
        self.probabilities_ = None
        self.tecs: List[TemporalEventCommunity] = []
        self._best_min_cluster_size = None

    def fit(
        self, events: List[Dict], contrastive_embeddings: np.ndarray
    ) -> List[TemporalEventCommunity]:
        n = len(events)
        if n < 3:
            logger.warning(f"Only {n} events, too few for clustering.")
            return self._single_cluster_fallback(events, contrastive_embeddings)

        timestamps = [e.get("timestamp", "static") for e in events]
        fourier_features = fourier_timestamp_encoding(
            timestamps, dim=self.config.timestamp_fourier_dim
        )

        combined = np.concatenate(
            [contrastive_embeddings, fourier_features], axis=1
        ).astype(np.float32)

        logger.info(
            f"ATEC: Combined feature dim = {combined.shape[1]} "
            f"(contrastive: {contrastive_embeddings.shape[1]}, "
            f"Fourier: {fourier_features.shape[1]})"
        )

        if self.config.auto_tune and n >= 20:
            best_size, best_score = self._auto_tune_min_cluster_size(combined)
            logger.info(
                f"ATEC: Auto-tuned min_cluster_size={best_size} "
                f"(validity score={best_score:.4f})"
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
        n_noise = np.sum(self.labels_ == -1)
        logger.info(
            f"ATEC: Found {n_clusters} clusters, {n_noise} noise points"
        )

        if self.config.noise_reassignment and n_noise > 0 and n_clusters > 0:
            self._reassign_noise(combined, events)

        self.tecs = self._build_tecs(events, contrastive_embeddings)

        logger.info(f"ATEC: Created {len(self.tecs)} Temporal Event Communities")
        return self.tecs

    def _auto_tune_min_cluster_size(
        self, features: np.ndarray
    ) -> Tuple[int, float]:
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
                )
                clusterer.fit(features)

                labels = clusterer.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                if n_clusters < 2:
                    continue

                if hasattr(clusterer, "relative_validity_"):
                    score = clusterer.relative_validity_
                else:
                    noise_ratio = np.sum(labels == -1) / len(labels)
                    score = (1.0 - noise_ratio) * min(n_clusters / 10.0, 1.0)

                if score > best_score:
                    best_score = score
                    best_size = size
            except Exception as e:
                logger.warning(f"ATEC auto-tune: size={size} failed: {e}")
                continue

        return best_size, best_score

    def _reassign_noise(self, features: np.ndarray, events: List[Dict]):
        noise_mask = self.labels_ == -1
        noise_indices = np.where(noise_mask)[0]

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

        logger.info(
            f"ATEC: Reassigned {len(noise_indices)} noise points to nearest clusters"
        )

    def _build_tecs(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> List[TemporalEventCommunity]:
        unique_labels = sorted(set(self.labels_))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        tecs = []
        for label in unique_labels:
            mask = self.labels_ == label
            indices = np.where(mask)[0].tolist()

            event_ids = []
            timestamps_in_cluster = []
            confidence_scores = {}

            for idx in indices:
                event_ids.append(events[idx].get("event_id", f"event-{idx}"))
                ts = events[idx].get("timestamp", "static")
                if ts and ts != "static":
                    timestamps_in_cluster.append(ts)
                confidence_scores[idx] = float(self.probabilities_[idx])

            centroid = embeddings[mask].mean(axis=0)

            earliest = min(timestamps_in_cluster) if timestamps_in_cluster else None
            latest = max(timestamps_in_cluster) if timestamps_in_cluster else None

            tec = TemporalEventCommunity(
                cluster_id=label,
                event_indices=indices,
                event_ids=event_ids,
                centroid=centroid,
                confidence_scores=confidence_scores,
                temporal_span=(earliest, latest),
            )
            tecs.append(tec)

        return tecs

    def _single_cluster_fallback(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> List[TemporalEventCommunity]:
        indices = list(range(len(events)))
        event_ids = [e.get("event_id", f"event-{i}") for i, e in enumerate(events)]
        timestamps = [
            e.get("timestamp", "static")
            for e in events
            if e.get("timestamp", "static") != "static"
        ]

        self.labels_ = np.zeros(len(events), dtype=int)
        self.probabilities_ = np.ones(len(events), dtype=float)

        tec = TemporalEventCommunity(
            cluster_id=0,
            event_indices=indices,
            event_ids=event_ids,
            centroid=embeddings.mean(axis=0),
            confidence_scores={i: 1.0 for i in indices},
            temporal_span=(
                min(timestamps) if timestamps else None,
                max(timestamps) if timestamps else None,
            ),
        )
        self.tecs = [tec]
        return self.tecs

    def get_event_cluster(self, event_index: int) -> Optional[int]:
        if self.labels_ is not None and event_index < len(self.labels_):
            return int(self.labels_[event_index])
        return None

    def get_event_confidence(self, event_index: int) -> float:
        if self.probabilities_ is not None and event_index < len(self.probabilities_):
            return float(self.probabilities_[event_index])
        return 0.0

    def get_cluster_by_id(self, cluster_id: int) -> Optional[TemporalEventCommunity]:
        for tec in self.tecs:
            if tec.cluster_id == cluster_id:
                return tec
        return None
