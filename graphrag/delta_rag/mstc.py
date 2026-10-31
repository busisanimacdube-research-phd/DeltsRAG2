import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("DELTA-RAG")


@dataclass
class MSTCConfig:
    projection_dim: int = 256
    hidden_dim: int = 512
    temperature: float = 0.07
    learning_rate: float = 1e-4
    num_epochs: int = 15
    batch_size: int = 64
    weight_decay: float = 1e-5

    macro_window: int = 365
    meso_window: int = 90
    micro_window: int = 30

    lambda_macro: float = 0.3
    lambda_meso: float = 0.4
    lambda_micro: float = 0.3

    hard_negative_min_days: int = 1825
    num_negatives: int = 7
    num_hard_negatives: int = 3

    min_events_for_training: int = 50
    min_triplets_for_training: int = 10


def _parse_timestamp_to_days(timestamp: str) -> Optional[int]:
    from .temporal_index import day_index
    return day_index(timestamp)


class ProjectionHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    neg_sim = F.cosine_similarity(
        anchor.unsqueeze(1), negatives, dim=-1
    ) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


class MultiScalePairMiner:

    def __init__(self, config: MSTCConfig):
        self.config = config
        self.scales = {
            "macro": config.macro_window,
            "meso": config.meso_window,
            "micro": config.micro_window,
        }

    def mine_pairs_at_scale(
        self,
        events: List[Dict],
        scale_name: str,
        window_days: int,
    ) -> List[Tuple[int, int, List[int]]]:
        n = len(events)
        if n < 3:
            return []

        entity_index: Dict[str, List[int]] = {}
        timestamps = []
        for i, event in enumerate(events):
            ts = _parse_timestamp_to_days(event.get("timestamp", "static"))
            timestamps.append(ts)
            for ent in event.get("entities_involved", []):
                entity_index.setdefault(ent.lower(), []).append(i)

        triplets = []
        all_indices = list(range(n))

        for anchor_idx in range(n):
            if timestamps[anchor_idx] is None:
                continue

            anchor_entities = set(
                e.lower() for e in events[anchor_idx].get("entities_involved", [])
            )
            if not anchor_entities:
                continue

            shared_entity_indices = set()
            for ent in anchor_entities:
                shared_entity_indices.update(entity_index.get(ent, []))
            shared_entity_indices.discard(anchor_idx)

            positives = []
            hard_negatives = []

            for idx in shared_entity_indices:
                if timestamps[idx] is None:
                    continue
                day_diff = abs(timestamps[anchor_idx] - timestamps[idx])
                if day_diff <= window_days:
                    positives.append(idx)
                elif day_diff >= self.config.hard_negative_min_days:
                    hard_negatives.append(idx)

            if not positives:
                continue

            pos_idx = random.choice(positives)

            neg_indices = []
            if hard_negatives:
                n_hard = min(self.config.num_hard_negatives, len(hard_negatives))
                neg_indices.extend(random.sample(hard_negatives, n_hard))

            exclude = {anchor_idx, pos_idx} | shared_entity_indices
            random_pool = [i for i in all_indices if i not in exclude]
            n_random = self.config.num_negatives - len(neg_indices)
            if random_pool and n_random > 0:
                n_random = min(n_random, len(random_pool))
                neg_indices.extend(random.sample(random_pool, n_random))

            if neg_indices:
                triplets.append((anchor_idx, pos_idx, neg_indices))

        return triplets

    def mine_all_scales(
        self, events: List[Dict]
    ) -> Dict[str, List[Tuple[int, int, List[int]]]]:
        result = {}
        for scale_name, window_days in self.scales.items():
            triplets = self.mine_pairs_at_scale(events, scale_name, window_days)
            result[scale_name] = triplets
            logger.info(
                f"MSTC: {scale_name} scale (window={window_days}d): "
                f"{len(triplets)} triplets mined"
            )
        return result


class ScaleTripletDataset(Dataset):

    def __init__(
        self,
        embeddings: np.ndarray,
        triplets: List[Tuple[int, int, List[int]]],
        num_negatives: int = 7,
    ):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.triplets = triplets
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_indices = self.triplets[idx]
        anchor = self.embeddings[anchor_idx]
        positive = self.embeddings[pos_idx]

        if len(neg_indices) < self.num_negatives:
            neg_indices = neg_indices + [neg_indices[-1]] * (
                self.num_negatives - len(neg_indices)
            )
        else:
            neg_indices = neg_indices[: self.num_negatives]

        negatives = self.embeddings[neg_indices]
        return anchor, positive, negatives


class MultiScaleTemporalContrastive:

    def __init__(self, config: MSTCConfig, input_dim: int = 1024):
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim,
        ).to(self.device)
        self.miner = MultiScalePairMiner(config)
        self._trained = False

    def fit_transform(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> np.ndarray:
        if len(events) < self.config.min_events_for_training:
            logger.warning(
                f"MSTC: Only {len(events)} events, below minimum "
                f"{self.config.min_events_for_training}. "
                "Returning L2-normalized base embeddings."
            )
            return self._normalize(embeddings)

        logger.info(f"MSTC: Mining multi-scale pairs from {len(events)} events...")
        scale_triplets = self.miner.mine_all_scales(events)

        total_triplets = sum(len(t) for t in scale_triplets.values())
        if total_triplets < self.config.min_triplets_for_training:
            logger.warning(
                f"MSTC: Only {total_triplets} total triplets across all scales. "
                "Returning normalized base embeddings."
            )
            return self._normalize(embeddings)

        logger.info(
            f"MSTC: Training with {total_triplets} triplets across 3 scales..."
        )
        self._train_multi_scale(embeddings, scale_triplets)

        logger.info("MSTC: Generating refined embeddings...")
        return self._transform(embeddings)

    def _train_multi_scale(
        self,
        embeddings: np.ndarray,
        scale_triplets: Dict[str, List[Tuple]],
    ):
        scale_weights = {
            "macro": self.config.lambda_macro,
            "meso": self.config.lambda_meso,
            "micro": self.config.lambda_micro,
        }

        scale_loaders = {}
        for scale_name, triplets in scale_triplets.items():
            if not triplets:
                continue
            dataset = ScaleTripletDataset(
                embeddings, triplets, self.config.num_negatives
            )
            scale_loaders[scale_name] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=len(dataset) > self.config.batch_size,
            )

        if not scale_loaders:
            logger.warning("MSTC: No valid scale dataloaders. Skipping training.")
            return

        optimizer = torch.optim.AdamW(
            self.projection_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = max(len(dl) for dl in scale_loaders.values()) * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        self.projection_head.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {s: 0.0 for s in scale_loaders}
            epoch_counts = {s: 0 for s in scale_loaders}
            total_loss_epoch = 0.0

            scale_iters = {
                name: iter(loader) for name, loader in scale_loaders.items()
            }

            max_batches = max(len(dl) for dl in scale_loaders.values())

            for batch_idx in range(max_batches):
                optimizer.zero_grad()
                combined_loss = torch.tensor(0.0, device=self.device)

                for scale_name, loader in scale_loaders.items():
                    try:
                        anchor, positive, negatives = next(scale_iters[scale_name])
                    except StopIteration:
                        scale_iters[scale_name] = iter(loader)
                        anchor, positive, negatives = next(scale_iters[scale_name])

                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negatives = negatives.to(self.device)

                    anchor_proj = self.projection_head(anchor)
                    positive_proj = self.projection_head(positive)

                    B, N, D = negatives.shape
                    neg_flat = negatives.view(B * N, D)
                    neg_proj = self.projection_head(neg_flat).view(B, N, -1)

                    scale_loss = info_nce_loss(
                        anchor_proj, positive_proj, neg_proj,
                        self.config.temperature,
                    )

                    weighted_loss = scale_weights[scale_name] * scale_loss
                    combined_loss = combined_loss + weighted_loss

                    epoch_losses[scale_name] += scale_loss.item()
                    epoch_counts[scale_name] += 1

                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                total_loss_epoch += combined_loss.item()

            if (epoch + 1) % max(1, self.config.num_epochs // 5) == 0:
                avg_total = total_loss_epoch / max(max_batches, 1)
                scale_avgs = {
                    s: epoch_losses[s] / max(epoch_counts[s], 1)
                    for s in epoch_losses
                }
                logger.info(
                    f"  MSTC Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Total: {avg_total:.4f}, "
                    f"Macro: {scale_avgs.get('macro', 0):.4f}, "
                    f"Meso: {scale_avgs.get('meso', 0):.4f}, "
                    f"Micro: {scale_avgs.get('micro', 0):.4f}"
                )

        self._trained = True

    @torch.no_grad()
    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.projection_head.eval()
        all_embeddings = torch.from_numpy(embeddings).float().to(self.device)

        batch_size = 256
        results = []
        for i in range(0, len(all_embeddings), batch_size):
            batch = all_embeddings[i : i + batch_size]
            projected = self.projection_head(batch)
            results.append(projected.cpu().numpy())

        return np.concatenate(results, axis=0)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def save(self, path: str):
        torch.save(
            {
                "projection_head": self.projection_head.state_dict(),
                "config": self.config,
                "input_dim": self.input_dim,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.projection_head.load_state_dict(checkpoint["projection_head"])
        self._trained = True
