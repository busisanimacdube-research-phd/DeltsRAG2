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

logger = logging.getLogger("CAST-RAG")


@dataclass
class ContrastiveConfig:
    projection_dim: int = 256
    hidden_dim: int = 512
    temperature: float = 0.07
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 64
    temporal_proximity_days: int = 365
    hard_negative_min_days: int = 1825
    num_negatives: int = 7
    num_hard_negatives: int = 3
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    min_events_for_training: int = 50


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


def _parse_timestamp_to_days(timestamp: str) -> Optional[int]:
    if not timestamp or timestamp == "static":
        return None
    try:
        parts = timestamp.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return year * 365 + month * 30 + day
    except (ValueError, IndexError):
        return None


class ContrastivePairMiner:

    def __init__(self, config: ContrastiveConfig):
        self.config = config

    def mine_pairs(
        self, events: List[Dict], embeddings: np.ndarray
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
                if day_diff <= self.config.temporal_proximity_days:
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


class ContrastiveTripletDataset(Dataset):

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


class ContrastiveTemporalEmbedding:

    def __init__(self, config: ContrastiveConfig, input_dim: int = 1024):
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim,
        ).to(self.device)
        self.miner = ContrastivePairMiner(config)
        self._trained = False

    def fit_transform(
        self, events: List[Dict], embeddings: np.ndarray
    ) -> np.ndarray:
        if len(events) < self.config.min_events_for_training:
            logger.warning(
                f"Only {len(events)} events, below minimum {self.config.min_events_for_training}. "
                "Returning L2-normalized base embeddings."
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms

        logger.info(f"Mining contrastive pairs from {len(events)} events...")
        triplets = self.miner.mine_pairs(events, embeddings)

        if len(triplets) < 10:
            logger.warning(
                f"Only {len(triplets)} triplets mined. Returning normalized base embeddings."
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms

        logger.info(f"Training contrastive projection with {len(triplets)} triplets...")
        self._train(embeddings, triplets)

        logger.info("Generating refined embeddings...")
        return self._transform(embeddings)

    def _train(self, embeddings: np.ndarray, triplets: List[Tuple]):
        dataset = ContrastiveTripletDataset(
            embeddings, triplets, self.config.num_negatives
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=len(dataset) > self.config.batch_size,
        )

        optimizer = torch.optim.AdamW(
            self.projection_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(dataloader) * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        self.projection_head.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for anchor, positive, negatives in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negatives = negatives.to(self.device)

                anchor_proj = self.projection_head(anchor)
                positive_proj = self.projection_head(positive)

                B, N, D = negatives.shape
                neg_flat = negatives.view(B * N, D)
                neg_proj = self.projection_head(neg_flat).view(B, N, -1)

                loss = info_nce_loss(
                    anchor_proj, positive_proj, neg_proj, self.config.temperature
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % max(1, self.config.num_epochs // 5) == 0:
                logger.info(
                    f"  CTE Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}"
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
        checkpoint = torch.load(path, map_location=self.device)
        self.projection_head.load_state_dict(checkpoint["projection_head"])
        self._trained = True
