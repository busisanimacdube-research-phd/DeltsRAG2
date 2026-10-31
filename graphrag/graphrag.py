import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast, Tuple
import torch
import time
import tiktoken
import glob
import json
import re
import networkx as nx
import copy
import random
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from nano_vectordb import NanoVectorDB
import numpy as np
from copy import deepcopy

from .prompt import PROMPTS, GRAPH_FIELD_SEP


from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
    bge_embedding,
    qwen_14b_complete,
    create_vllm_complete_function,
)
from ._op import (
    chunking_by_token_size,
    extract_events,
    get_chunks,
    normalize_timestamp,
)

from .contrastive import ContrastiveTemporalEmbedding, ContrastiveConfig
from .clustering import AdaptiveTemporalClustering, ClusteringConfig, TemporalEventCommunity
from .bridging import SemanticEventBridging, BridgingConfig
from .hierarchical_retrieval import (
    ConfidenceWeightedHierarchicalRetrieval,
    HierarchicalRetrievalConfig,
)

from .delta_rag import (
    MultiScaleTemporalContrastive,
    MSTCConfig,
    DensityAdaptiveTemporalPartitioning,
    DATPConfig,
    TemporalKnowledgePartition,
    CrossPartitionKGBridging,
    CPKBConfig,
    AdaptiveHierarchicalRetrieval,
    AHKRConfig,
)

from ._storage import (
    JsonKVStorage,
    NetworkXStorage,
    NanoVectorDBStorage,
    TimestampEnhancedVectorStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    compute_args_hash,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available. Cross-encoder reranking will be disabled.")


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./dyg_rag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 64
    tiktoken_model_name: str = "cl100k_base"

    event_extract_max_gleaning: int = 1
    if_wri_ents: bool = False

    ent_factor: float = 0.3
    ent_ratio: float = 0.75
    time_ratio: float = 0.25
    decay_rate: float = 0.02
    time_weight: float = 0.1
    time_factor: float = 1.0
    max_links: int = 6

    enable_graph_traversal: bool = True
    walk_depth: int = 2
    walk_nodes: int = 5
    walk_n: int = 3

    enable_bm25_reranking: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_weight: float = 0.7
    entity_match_weight: float = 0.3

    enable_ce_rerank: bool = True
    ce_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    ce_max_len: int = 512
    ce_batch_size: int = 16
    ce_device: str = "auto"
    ce_weight: float = 0.8
    ce_ent_weight: float = 0.2

    ce_cache_size: int = 1000
    ce_truncate_len: int = 400
    ce_dynamic_batch: bool = True
    ce_min_batch: int = 8
    ce_max_batch: int = 64
    ce_early_stop: float = 0.9

    model_path: str = "./models"

    ner_model_name: str = "dslim_bert_base_ner"
    ner_device: str = "cuda:0"
    ner_batch_size: int = 32

    enable_fuzzy_entity_matching: bool = True
    fuzzy_match_threshold: float = 0.8


    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 32
    query_better_than_threshold: float = 0.2

    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = ""
    cheap_model_id: str = ""
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16


    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    enable_timestamp_encoding: bool = True
    timestamp_dim: int = 16

    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    if_timeline_events: bool = True

    random_seed: int = 42


    enable_cast_rag: bool = True

    enable_cte: bool = True
    cte_projection_dim: int = 256
    cte_hidden_dim: int = 512
    cte_temperature: float = 0.07
    cte_learning_rate: float = 1e-4
    cte_num_epochs: int = 10
    cte_batch_size: int = 64
    cte_temporal_proximity_days: int = 365
    cte_hard_negative_min_days: int = 1825
    cte_num_negatives: int = 7
    cte_num_hard_negatives: int = 3

    enable_atec: bool = True
    atec_min_cluster_sizes: tuple = (3, 5, 7, 10)
    atec_min_samples: int = 2
    atec_auto_tune: bool = True
    atec_timestamp_fourier_dim: int = 16

    enable_seb: bool = True
    seb_similarity_threshold: float = 0.6
    seb_top_k_bridges: int = 3
    seb_max_bridges: int = 100
    seb_temporal_decay_rate: float = 0.005
    seb_entity_priority_boost: float = 1.5

    enable_cwhr: bool = True
    cwhr_top_c_clusters: int = 5
    cwhr_alpha: float = 0.4
    cwhr_beta: float = 0.25
    cwhr_gamma: float = 0.2
    cwhr_delta: float = 0.15


    enable_delta_rag: bool = False

    enable_mstc: bool = True
    mstc_projection_dim: int = 256
    mstc_hidden_dim: int = 512
    mstc_temperature: float = 0.07
    mstc_learning_rate: float = 1e-4
    mstc_num_epochs: int = 15
    mstc_batch_size: int = 64
    mstc_macro_window: int = 365
    mstc_meso_window: int = 90
    mstc_micro_window: int = 30
    mstc_lambda_macro: float = 0.3
    mstc_lambda_meso: float = 0.4
    mstc_lambda_micro: float = 0.3
    mstc_hard_negative_min_days: int = 1825
    mstc_num_negatives: int = 7
    mstc_num_hard_negatives: int = 3

    enable_datp: bool = True
    datp_min_cluster_sizes: tuple = (3, 5, 7, 10)
    datp_min_samples: int = 2
    datp_auto_tune: bool = True
    datp_fourier_dim: int = 16

    enable_cpkb: bool = True
    cpkb_similarity_threshold: float = 0.6
    cpkb_top_k_bridges: int = 3
    cpkb_max_bridges: int = 100
    cpkb_temporal_decay_rate: float = 0.005
    cpkb_entity_boost: float = 1.5

    enable_ahkr: bool = True
    ahkr_top_p_partitions: int = 5
    ahkr_alpha: float = 0.35
    ahkr_beta: float = 0.20
    ahkr_gamma: float = 0.15
    ahkr_delta: float = 0.10
    ahkr_epsilon: float = 0.20
    ahkr_temporal_boost: float = 1.5
    ahkr_entity_boost: float = 1.3

    enable_entity_anchor: bool = False
    entity_anchor_top_m: int = 20
    entity_anchor_matching: str = "strict"
    answer_commit: bool = False
    enable_conflict_fusion: bool = False
    source_credibility: dict = field(default_factory=dict)
    event_hook: object = None

    enable_arbiter: bool = False
    arbiter_require_conflict: bool = False
    arbiter_max_slots: int = 8
    arbiter_max_unslotted_events: int = 40
    arbiter_open_ended_penalty: float = 0.6
    arbiter_min_holder_score: float = 0.25
    arbiter_enable_interval_recovery: bool = True
    arbiter_enable_terminator_closure: bool = True
    arbiter_enable_supersession_closure: bool = True
    arbiter_enable_corroboration: bool = True
    arbiter_retain_alternatives: bool = True
    arbiter_show_arbitration: bool = True

    def get_config_dict(self):
        config = {}

        for k, v in self.__dict__.items():
            config[k] = v

        keys_to_exclude = [
            "event_dynamic_graph",
            "events_vdb"
            "full_docs",
            "text_chunks",
            "llm_response_cache",
            "_arbiter",
            "_last_arbiter_stats",
        ]

        for key in keys_to_exclude:
            if key in config:
                config.pop(key)

        return config

    def __post_init__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self._original_vector_db_storage_cls = self.vector_db_storage_cls

        if self.enable_timestamp_encoding:
            self.vector_db_storage_cls = TimestampEnhancedVectorStorage
            if "timestamp_dim" not in self.vector_db_storage_cls_kwargs:
                self.vector_db_storage_cls_kwargs["timestamp_dim"] = self.timestamp_dim
            logger.info(
                f"Enabled timestamp encoding with dimension {self.timestamp_dim}"
            )

        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(
                self.best_model_id
            )
            self.cheap_model_func = create_amazon_bedrock_complete_function(
                self.cheap_model_id
            )
            self.embedding_func = amazon_bedrock_embedding
            logger.info("Switched the default openai funcs to Amazon Bedrock")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
            
        config_dict = self.get_config_dict()

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=config_dict
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=config_dict
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=config_dict
            )
            if self.enable_llm_cache
            else None
        )

        self.event_dynamic_graph = self.graph_storage_cls(
            namespace="event_dynamic_graph", global_config=config_dict
        )
        
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        self.events_vdb = (
            self.vector_db_storage_cls(
                namespace="events",
                global_config=config_dict,
                embedding_func=self.embedding_func,
                meta_fields={"event_id", "timestamp", "sentence", "source_id", "context"},
            )
        )
        

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        self.cross_encoder = None
        if self.enable_ce_rerank:
            if not CROSSENCODER_AVAILABLE:
                logger.warning(
                    "Cross-encoder reranking is enabled but sentence-transformers is not available. Falling back to BM25."
                )
                self.enable_ce_rerank = False
                self.enable_bm25_reranking = True
            else:
                try:
                    model_name = self.ce_model

                    local_model_path = (
                        Path(self.model_path) / model_name.replace("/", "_")
                    )
                    model_path = model_name

                    if local_model_path.exists():
                        config_file = local_model_path / "config.json"
                        model_files = list(
                            local_model_path.glob("*.bin")
                        ) + list(local_model_path.glob("*.safetensors")) + list(local_model_path.glob("pytorch_model.bin"))

                        if config_file.exists() and (
                            model_files
                            or any(local_model_path.glob("model.safetensors"))
                        ):
                            model_path = str(local_model_path)
                            logger.info(
                                f"Loading cross-encoder from validated local path: {model_path}"
                            )
                        else:
                            logger.warning(
                                f"Local model path exists but missing required files (config.json or model files). Using original model name: {model_name}"
                            )
                    else:
                        logger.info(
                            f"Local model path not found. Using original model name: {model_name}"
                        )

                    device = self.ce_device
                    if device == "auto":
                        try:
                            import torch

                            device = "cuda" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            device = "cpu"

                    self.cross_encoder = CrossEncoder(
                        model_name=model_path,
                        max_length=self.ce_max_len,
                        device=device,
                    )
                    logger.info(
                        f"Cross-encoder initialized successfully on device: {device}"
                    )

                    self.cross_encoder_cache = {}
                    logger.info(
                        f"Cross-encoder optimization enabled with cache size: {self.ce_cache_size}"
                    )

                    if self.enable_bm25_reranking:
                        logger.info(
                            "Cross-encoder enabled. BM25 reranking will be disabled unless explicitly kept enabled."
                        )
                        self.enable_bm25_reranking = False

                except Exception as e:
                    logger.error(
                        f"Failed to initialize cross-encoder model {model_name}: {e}"
                    )
                    logger.info("Falling back to BM25 reranking")
                    self.enable_ce_rerank = False
                    self.enable_bm25_reranking = True
                    self.cross_encoder = None

        self._cast_rag_events = []
        self._cast_rag_embeddings = None
        self._cast_rag_tecs = []
        self._cast_rag_cte = None
        self._cast_rag_atec = None
        self._cast_rag_seb = None
        self._cast_rag_cwhr = None

        if self.enable_delta_rag and self.enable_cast_rag:
            self.enable_cast_rag = False
            logger.info("DELTA-RAG enabled — CAST-RAG disabled (mutually exclusive)")

        if self.enable_cast_rag:
            self._init_cast_rag_modules()
            logger.info("CAST-RAG modules initialized")

        self._delta_rag_events = []
        self._delta_rag_embeddings = None
        self._delta_rag_tkps = []
        self._delta_rag_mstc = None
        self._delta_rag_datp = None
        self._delta_rag_cpkb = None
        self._delta_rag_ahkr = None

        self._entity_event_index = None
        self._conflict_index = None
        self.last_conflict_stats = {}

        self._arbiter = None
        self._last_arbiter_stats = None
        if self.enable_arbiter:
            self._init_arbiter()
            logger.info("ARBITER arbitration layer initialized")

        if self.enable_delta_rag:
            self._init_delta_rag_modules()
            logger.info("DELTA-RAG modules initialized")
            if self._load_delta_rag_state():
                logger.info(
                    f"DELTA-RAG state loaded: {len(self._delta_rag_events)} events, "
                    f"{len(self._delta_rag_tkps)} TKPs"
                )

        self._log_event_graph_stats()

        self._log_storage_overview()

    def _init_cast_rag_modules(self):
        if self.enable_cte:
            cte_config = ContrastiveConfig(
                projection_dim=self.cte_projection_dim,
                hidden_dim=self.cte_hidden_dim,
                temperature=self.cte_temperature,
                learning_rate=self.cte_learning_rate,
                num_epochs=self.cte_num_epochs,
                batch_size=self.cte_batch_size,
                temporal_proximity_days=self.cte_temporal_proximity_days,
                hard_negative_min_days=self.cte_hard_negative_min_days,
                num_negatives=self.cte_num_negatives,
                num_hard_negatives=self.cte_num_hard_negatives,
            )
            self._cast_rag_cte_config = cte_config

        if self.enable_atec:
            atec_config = ClusteringConfig(
                min_cluster_size_range=self.atec_min_cluster_sizes,
                min_samples=self.atec_min_samples,
                auto_tune=self.atec_auto_tune,
                timestamp_fourier_dim=self.atec_timestamp_fourier_dim,
            )
            self._cast_rag_atec = AdaptiveTemporalClustering(atec_config)

        if self.enable_seb:
            seb_config = BridgingConfig(
                similarity_threshold=self.seb_similarity_threshold,
                top_k_bridges_per_cluster_pair=self.seb_top_k_bridges,
                max_total_bridges=self.seb_max_bridges,
                temporal_decay_rate=self.seb_temporal_decay_rate,
                entity_priority_boost=self.seb_entity_priority_boost,
            )
            self._cast_rag_seb = SemanticEventBridging(seb_config)

        if self.enable_cwhr:
            cwhr_config = HierarchicalRetrievalConfig(
                top_c_clusters=self.cwhr_top_c_clusters,
                alpha=self.cwhr_alpha,
                beta=self.cwhr_beta,
                gamma=self.cwhr_gamma,
                delta=self.cwhr_delta,
            )
            self._cast_rag_cwhr = ConfidenceWeightedHierarchicalRetrieval(cwhr_config)

    async def _run_cast_rag_pipeline(self, all_events_data: list):
        if not self.enable_cast_rag or not all_events_data:
            return

        logger.info(f"=== CAST-RAG Pipeline: Processing {len(all_events_data)} events ===")

        events = []
        for ev in all_events_data:
            events.append({
                "event_id": ev.get("event_id", ""),
                "timestamp": ev.get("timestamp", "static"),
                "sentence": ev.get("sentence", ""),
                "context": ev.get("context", ""),
                "entities_involved": ev.get("entities_involved", []),
                "source_id": ev.get("source_id", ""),
            })

        logger.info("CAST-RAG: Computing base embeddings for events...")
        event_texts = []
        for ev in events:
            text = ev["sentence"]
            if ev["timestamp"] != "static":
                text += f" (Time: {ev['timestamp']})"
            event_texts.append(text)

        base_embeddings = []
        batch_size = self.embedding_batch_num
        for i in range(0, len(event_texts), batch_size):
            batch = event_texts[i:i + batch_size]
            batch_emb = await self.embedding_func(batch)
            base_embeddings.append(batch_emb)

        base_embeddings = np.concatenate(base_embeddings, axis=0)
        input_dim = base_embeddings.shape[1]
        logger.info(f"CAST-RAG: Base embeddings shape: {base_embeddings.shape}")

        if self.enable_cte:
            logger.info("CAST-RAG Step 1: Contrastive Temporal Event Embedding (CTE)")
            self._cast_rag_cte = ContrastiveTemporalEmbedding(
                self._cast_rag_cte_config, input_dim=input_dim
            )
            contrastive_embeddings = self._cast_rag_cte.fit_transform(events, base_embeddings)
            logger.info(f"CAST-RAG: Contrastive embeddings shape: {contrastive_embeddings.shape}")
        else:
            contrastive_embeddings = base_embeddings
            norms = np.linalg.norm(contrastive_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            contrastive_embeddings = contrastive_embeddings / norms

        tecs = []
        if self.enable_atec and self._cast_rag_atec is not None:
            logger.info("CAST-RAG Step 2: Adaptive Temporal Event Clustering (ATEC)")
            tecs = self._cast_rag_atec.fit(events, contrastive_embeddings)
            logger.info(f"CAST-RAG: Created {len(tecs)} TECs")
        else:
            from .clustering import AdaptiveTemporalClustering, ClusteringConfig
            fallback = AdaptiveTemporalClustering(ClusteringConfig())
            tecs = fallback._single_cluster_fallback(events, contrastive_embeddings)

        if self.enable_seb and self._cast_rag_seb is not None and len(tecs) >= 2:
            logger.info("CAST-RAG Step 3: Semantic Event Bridging (SEB)")
            bridges = self._cast_rag_seb.compute_bridges(tecs, events, contrastive_embeddings)

            if bridges and self.event_dynamic_graph:
                for bridge in bridges:
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
                    try:
                        await self.event_dynamic_graph.upsert_edge(src_id, tgt_id, edge_data)
                        await self.event_dynamic_graph.upsert_edge(tgt_id, src_id, edge_data)
                    except Exception as e:
                        logger.debug(f"Could not add bridge edge {src_id}->{tgt_id}: {e}")

                logger.info(f"CAST-RAG: Added {len(bridges)} bridge edges to graph")

        self._cast_rag_events = events
        self._cast_rag_embeddings = contrastive_embeddings
        self._cast_rag_tecs = tecs

        self._save_cast_rag_state()

        logger.info("=== CAST-RAG Pipeline Complete ===")

    def _save_cast_rag_state(self):
        try:
            state_dir = os.path.join(self.working_dir, "cast_rag_state")
            os.makedirs(state_dir, exist_ok=True)

            if self._cast_rag_embeddings is not None:
                np.save(os.path.join(state_dir, "embeddings.npy"), self._cast_rag_embeddings)

            if self._cast_rag_events:
                import json as json_module
                with open(os.path.join(state_dir, "events.json"), "w") as f:
                    json_module.dump(self._cast_rag_events, f)

            if self._cast_rag_tecs:
                tec_data = []
                for tec in self._cast_rag_tecs:
                    tec_data.append({
                        "cluster_id": tec.cluster_id,
                        "event_indices": tec.event_indices,
                        "event_ids": tec.event_ids,
                        "confidence_scores": {str(k): v for k, v in tec.confidence_scores.items()},
                        "temporal_span": list(tec.temporal_span),
                        "avg_confidence": tec.avg_confidence,
                    })
                import json as json_module
                with open(os.path.join(state_dir, "tecs.json"), "w") as f:
                    json_module.dump(tec_data, f)

            if self._cast_rag_cte is not None:
                self._cast_rag_cte.save(os.path.join(state_dir, "cte_model.pt"))

            logger.info(f"CAST-RAG state saved to {state_dir}")
        except Exception as e:
            logger.warning(f"Failed to save CAST-RAG state: {e}")

    def _load_cast_rag_state(self):
        state_dir = os.path.join(self.working_dir, "cast_rag_state")
        if not os.path.exists(state_dir):
            return False

        try:
            emb_path = os.path.join(state_dir, "embeddings.npy")
            events_path = os.path.join(state_dir, "events.json")
            tecs_path = os.path.join(state_dir, "tecs.json")

            if os.path.exists(emb_path):
                self._cast_rag_embeddings = np.load(emb_path)

            if os.path.exists(events_path):
                import json as json_module
                with open(events_path, "r") as f:
                    self._cast_rag_events = json_module.load(f)

            if os.path.exists(tecs_path):
                import json as json_module
                with open(tecs_path, "r") as f:
                    tec_data = json_module.load(f)

                self._cast_rag_tecs = []
                for td in tec_data:
                    centroid_idx = td["event_indices"]
                    centroid = self._cast_rag_embeddings[centroid_idx].mean(axis=0) if self._cast_rag_embeddings is not None else np.zeros(256)
                    tec = TemporalEventCommunity(
                        cluster_id=td["cluster_id"],
                        event_indices=td["event_indices"],
                        event_ids=td["event_ids"],
                        centroid=centroid,
                        confidence_scores={int(k): v for k, v in td["confidence_scores"].items()},
                        temporal_span=tuple(td["temporal_span"]),
                        avg_confidence=td.get("avg_confidence", 0.0),
                    )
                    self._cast_rag_tecs.append(tec)

            cte_path = os.path.join(state_dir, "cte_model.pt")
            if os.path.exists(cte_path) and self.enable_cte:
                input_dim = self._cast_rag_embeddings.shape[1] if self._cast_rag_embeddings is not None else 1024
                self._cast_rag_cte = ContrastiveTemporalEmbedding(
                    self._cast_rag_cte_config, input_dim=input_dim
                )
                self._cast_rag_cte.load(cte_path)

            logger.info(f"CAST-RAG state loaded from {state_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load CAST-RAG state: {e}")
            return False


    def _init_delta_rag_modules(self):
        if self.enable_mstc:
            self._delta_rag_mstc_config = MSTCConfig(
                projection_dim=self.mstc_projection_dim,
                hidden_dim=self.mstc_hidden_dim,
                temperature=self.mstc_temperature,
                learning_rate=self.mstc_learning_rate,
                num_epochs=self.mstc_num_epochs,
                batch_size=self.mstc_batch_size,
                macro_window=self.mstc_macro_window,
                meso_window=self.mstc_meso_window,
                micro_window=self.mstc_micro_window,
                lambda_macro=self.mstc_lambda_macro,
                lambda_meso=self.mstc_lambda_meso,
                lambda_micro=self.mstc_lambda_micro,
                hard_negative_min_days=self.mstc_hard_negative_min_days,
                num_negatives=self.mstc_num_negatives,
                num_hard_negatives=self.mstc_num_hard_negatives,
            )

        if self.enable_datp:
            datp_config = DATPConfig(
                min_cluster_size_range=self.datp_min_cluster_sizes,
                min_samples=self.datp_min_samples,
                auto_tune=self.datp_auto_tune,
                fourier_dim=self.datp_fourier_dim,
            )
            self._delta_rag_datp = DensityAdaptiveTemporalPartitioning(datp_config)

        if self.enable_cpkb:
            cpkb_config = CPKBConfig(
                similarity_threshold=self.cpkb_similarity_threshold,
                top_k_bridges_per_pair=self.cpkb_top_k_bridges,
                max_total_bridges=self.cpkb_max_bridges,
                temporal_decay_rate=self.cpkb_temporal_decay_rate,
                entity_boost=self.cpkb_entity_boost,
            )
            self._delta_rag_cpkb = CrossPartitionKGBridging(cpkb_config)

        if self.enable_ahkr:
            ahkr_config = AHKRConfig(
                top_p_partitions=self.ahkr_top_p_partitions,
                alpha=self.ahkr_alpha,
                beta=self.ahkr_beta,
                gamma=self.ahkr_gamma,
                delta=self.ahkr_delta,
                epsilon=self.ahkr_epsilon,
                temporal_boost=self.ahkr_temporal_boost,
                entity_boost=self.ahkr_entity_boost,
            )
            self._delta_rag_ahkr = AdaptiveHierarchicalRetrieval(ahkr_config)

    async def _run_delta_rag_pipeline(self, all_events_data: list):
        if not self.enable_delta_rag or not all_events_data:
            return

        logger.info(f"=== DELTA-RAG Pipeline: Processing {len(all_events_data)} events ===")

        if self.event_dynamic_graph:
            try:
                all_graph_nodes = await self.event_dynamic_graph.get_all_nodes()
                graph_node_count = len(all_graph_nodes) if all_graph_nodes else 0
                if graph_node_count > 0 and len(all_events_data) < graph_node_count:
                    logger.warning(
                        f"DELTA-RAG: Pipeline received {len(all_events_data)} events "
                        f"but graph has {graph_node_count} nodes. Reloading from graph."
                    )
                    reloaded = []
                    for nid, gdata in all_graph_nodes.items():
                        if gdata:
                            reloaded.append({
                                "event_id": nid,
                                "timestamp": gdata.get("timestamp", "static"),
                                "sentence": gdata.get("sentence", ""),
                                "context": gdata.get("context", ""),
                                "entities_involved": gdata.get("entities_involved", []),
                                "source_id": gdata.get("source_id", ""),
                            })
                    if len(reloaded) > len(all_events_data):
                        all_events_data = reloaded
                        logger.info(f"DELTA-RAG: Reloaded {len(all_events_data)} events from graph")
            except Exception as e:
                logger.debug(f"DELTA-RAG: Event count validation skipped: {e}")

        events = []
        for ev in all_events_data:
            events.append({
                "event_id": ev.get("event_id", ""),
                "timestamp": ev.get("timestamp", "static"),
                "sentence": ev.get("sentence", ""),
                "context": ev.get("context", ""),
                "entities_involved": ev.get("entities_involved", []),
                "source_id": ev.get("source_id", ""),
            })

        logger.info("DELTA-RAG: Computing base embeddings...")
        event_texts = []
        for ev in events:
            text = ev["sentence"]
            if ev["timestamp"] != "static":
                text += f" (Time: {ev['timestamp']})"
            event_texts.append(text)

        base_embeddings = []
        batch_size = self.embedding_batch_num
        for i in range(0, len(event_texts), batch_size):
            batch = event_texts[i:i + batch_size]
            batch_emb = await self.embedding_func(batch)
            base_embeddings.append(batch_emb)

        base_embeddings = np.concatenate(base_embeddings, axis=0)
        input_dim = base_embeddings.shape[1]
        logger.info(f"DELTA-RAG: Base embeddings shape: {base_embeddings.shape}")

        if self.enable_mstc:
            logger.info("DELTA-RAG Step 1: Multi-Scale Temporal Contrastive (MSTC)")
            self._delta_rag_mstc = MultiScaleTemporalContrastive(
                self._delta_rag_mstc_config, input_dim=input_dim
            )
            mstc_embeddings = self._delta_rag_mstc.fit_transform(events, base_embeddings)
            logger.info(f"DELTA-RAG: MSTC embeddings shape: {mstc_embeddings.shape}")
        else:
            mstc_embeddings = base_embeddings
            norms = np.linalg.norm(mstc_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            mstc_embeddings = mstc_embeddings / norms

        tkps = []
        if self.enable_datp and self._delta_rag_datp is not None:
            logger.info("DELTA-RAG Step 2: Density-Adaptive Temporal Partitioning (DATP)")
            tkps = self._delta_rag_datp.fit(events, mstc_embeddings)
            logger.info(f"DELTA-RAG: Created {len(tkps)} TKPs")
        else:
            fallback = DensityAdaptiveTemporalPartitioning(DATPConfig())
            tkps = fallback._single_partition_fallback(events, mstc_embeddings)

        if self.enable_cpkb and self._delta_rag_cpkb is not None and len(tkps) >= 2:
            logger.info("DELTA-RAG Step 3: Cross-Partition KG Bridging (CPKB)")
            bridges = self._delta_rag_cpkb.compute_bridges(tkps, events, mstc_embeddings)

            if bridges and self.event_dynamic_graph:
                added = await self._delta_rag_cpkb.add_bridges_to_graph(
                    self.event_dynamic_graph, events
                )
                logger.info(f"DELTA-RAG: Added {added} bridge edges to graph")

        self._delta_rag_events = events
        self._delta_rag_embeddings = mstc_embeddings
        self._delta_rag_tkps = tkps

        self._save_delta_rag_state()

        logger.info("=== DELTA-RAG Pipeline Complete ===")

    def _save_delta_rag_state(self):
        try:
            state_dir = os.path.join(self.working_dir, "delta_rag_state")
            os.makedirs(state_dir, exist_ok=True)

            if self._delta_rag_embeddings is not None:
                np.save(os.path.join(state_dir, "embeddings.npy"), self._delta_rag_embeddings)

            if self._delta_rag_events:
                import json as json_module
                with open(os.path.join(state_dir, "events.json"), "w") as f:
                    json_module.dump(self._delta_rag_events, f)

            if self._delta_rag_tkps:
                tkp_data = []
                for tkp in self._delta_rag_tkps:
                    tkp_data.append({
                        "partition_id": tkp.partition_id,
                        "event_indices": tkp.event_indices,
                        "event_ids": tkp.event_ids,
                        "confidence_scores": {str(k): v for k, v in tkp.confidence_scores.items()},
                        "temporal_span": list(tkp.temporal_span),
                        "avg_confidence": tkp.avg_confidence,
                    })
                import json as json_module
                with open(os.path.join(state_dir, "tkps.json"), "w") as f:
                    json_module.dump(tkp_data, f)

            if self._delta_rag_cpkb is not None and self._delta_rag_cpkb.bridges:
                import json as json_module
                with open(os.path.join(state_dir, "bridges.json"), "w") as f:
                    json_module.dump(self._delta_rag_cpkb.bridges, f)

            if self._delta_rag_mstc is not None:
                self._delta_rag_mstc.save(os.path.join(state_dir, "mstc_model.pt"))

            logger.info(f"DELTA-RAG state saved to {state_dir}")
        except Exception as e:
            logger.warning(f"Failed to save DELTA-RAG state: {e}")

    def _load_delta_rag_state(self):
        state_dir = os.path.join(self.working_dir, "delta_rag_state")
        if not os.path.exists(state_dir):
            return False

        try:
            emb_path = os.path.join(state_dir, "embeddings.npy")
            events_path = os.path.join(state_dir, "events.json")
            tkps_path = os.path.join(state_dir, "tkps.json")

            if os.path.exists(emb_path):
                self._delta_rag_embeddings = np.load(emb_path)

            if os.path.exists(events_path):
                import json as json_module
                with open(events_path, "r") as f:
                    self._delta_rag_events = json_module.load(f)

            if os.path.exists(tkps_path):
                import json as json_module
                with open(tkps_path, "r") as f:
                    tkp_data = json_module.load(f)

                self._delta_rag_tkps = []
                for td in tkp_data:
                    centroid_idx = td["event_indices"]
                    centroid = (
                        self._delta_rag_embeddings[centroid_idx].mean(axis=0)
                        if self._delta_rag_embeddings is not None
                        else np.zeros(256)
                    )
                    tkp = TemporalKnowledgePartition(
                        partition_id=td["partition_id"],
                        event_indices=td["event_indices"],
                        event_ids=td["event_ids"],
                        centroid=centroid,
                        confidence_scores={int(k): v for k, v in td["confidence_scores"].items()},
                        temporal_span=tuple(td["temporal_span"]),
                        avg_confidence=td.get("avg_confidence", 0.0),
                    )
                    self._delta_rag_tkps.append(tkp)

            bridges_path = os.path.join(state_dir, "bridges.json")
            if (self.enable_cpkb and self._delta_rag_cpkb is not None
                    and os.path.exists(bridges_path)):
                import json as json_module
                with open(bridges_path, "r") as f:
                    self._delta_rag_cpkb.bridges = json_module.load(f)
                logger.info(f"DELTA-RAG: loaded {len(self._delta_rag_cpkb.bridges)} CPKB bridges")

            mstc_path = os.path.join(state_dir, "mstc_model.pt")
            if os.path.exists(mstc_path) and self.enable_mstc:
                input_dim = (
                    self._delta_rag_embeddings.shape[1]
                    if self._delta_rag_embeddings is not None
                    else 1024
                )
                self._delta_rag_mstc = MultiScaleTemporalContrastive(
                    self._delta_rag_mstc_config, input_dim=input_dim
                )
                self._delta_rag_mstc.load(mstc_path)

            logger.info(f"DELTA-RAG state loaded from {state_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load DELTA-RAG state: {e}")
            return False

    def log_data_statistics(self):
        logger.info("=== CAST-RAG Data Statistics ===")
        self._log_event_graph_stats()
        self._log_storage_overview()
        if self._cast_rag_tecs:
            logger.info(f"CAST-RAG TECs: {len(self._cast_rag_tecs)} clusters")
            if self._cast_rag_seb and self._cast_rag_seb.bridges:
                logger.info(f"CAST-RAG Bridges: {len(self._cast_rag_seb.bridges)}")
        logger.info("=== End Data Statistics ===")

    def _log_event_graph_stats(self):
        try:
            if hasattr(self.event_dynamic_graph, '_graph') and self.event_dynamic_graph._graph:
                graph = self.event_dynamic_graph._graph
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                
                if num_nodes > 0:
                    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
                    
                    logger.info(f"Event Dynamic Graph loaded successfully:")
                    logger.info(f"  • Nodes (events): {num_nodes:,}")
                    logger.info(f"  • Edges (relationships): {num_edges:,}")
                    logger.info(f"  • Average node degree: {avg_degree:.2f}")
                    
                    if num_nodes > 1:
                        max_edges = num_nodes * (num_nodes - 1) // 2
                        density = num_edges / max_edges if max_edges > 0 else 0
                        logger.info(f"  • Graph density: {density:.4f}")
                else:
                    logger.info("Event Dynamic Graph is empty (no events loaded yet)")
            else:
                logger.warning("Event Dynamic Graph is not properly initialized")
                
            self._log_events_vdb_stats()
                
        except Exception as e:
            logger.warning(f"Could not retrieve event graph statistics: {e}")
            
    def _log_events_vdb_stats(self):
        try:
            if hasattr(self.events_vdb, '_client') and self.events_vdb._client:
                client = self.events_vdb._client
                
                doc_count = 0
                
                if hasattr(client, '__storage') and isinstance(client.__storage, dict):
                    storage = client.__storage
                    
                    if 'matrix' in storage and storage['matrix'] is not None:
                        if hasattr(storage['matrix'], 'shape'):
                            doc_count = storage['matrix'].shape[0]
                        elif isinstance(storage['matrix'], list):
                            doc_count = len(storage['matrix'])
                    elif 'data' in storage and storage['data'] is not None:
                        if isinstance(storage['data'], list):
                            doc_count = len(storage['data'])
                
                if doc_count == 0:
                    if hasattr(client, 'get_current_count'):
                        doc_count = client.get_current_count()
                    elif hasattr(client, 'count'):
                        doc_count = client.count()
                    elif hasattr(client, '__len__'):
                        doc_count = len(client)
                
                if doc_count > 0:
                    logger.info(f"Events Vector Database loaded successfully:")
                    logger.info(f"  • Documents (events): {doc_count:,}")
                    
                    if hasattr(client, 'vec_dim'):
                        logger.info(f"  • Embedding dimension: {client.vec_dim}")
                    elif hasattr(self.events_vdb, 'enhanced_dim'):
                        logger.info(f"  • Enhanced embedding dimension: {self.events_vdb.enhanced_dim}")
                        if hasattr(self.events_vdb, 'original_dim') and hasattr(self.events_vdb, 'timestamp_dim'):
                            logger.info(f"    - Semantic: {self.events_vdb.original_dim}, Timestamp: {self.events_vdb.timestamp_dim}")
                else:
                    logger.info("Events Vector Database is empty (no events indexed yet)")
            else:
                logger.warning("Events Vector Database is not properly initialized")
        except Exception as e:
            logger.warning(f"Could not retrieve events vector database statistics: {e}")

    def _log_storage_overview(self):
        try:
            logger.info("--- Storage Overview ---")
            
            if hasattr(self.full_docs, '_data'):
                logger.info(f"Full Docs Storage: {len(self.full_docs._data):,} documents")
            else:
                logger.info("Full Docs Storage: not initialized or unavailable")
                
            if hasattr(self.text_chunks, '_data'):
                logger.info(f"Text Chunks Storage: {len(self.text_chunks._data):,} chunks")
            else:
                logger.info("Text Chunks Storage: not initialized or unavailable")
                
            if self.llm_response_cache and hasattr(self.llm_response_cache, '_data'):
                logger.info(f"LLM Response Cache: {len(self.llm_response_cache._data):,} cached responses")
            else:
                logger.info("LLM Response Cache: disabled or unavailable")
                
            logger.info("--- End Storage Overview ---")
        except Exception as e:
            logger.warning(f"Could not retrieve storage overview: {e}")

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def _normalize_entity(self, s: str) -> str:
        from .delta_rag import entity_matching as em
        return em.normalize_v1(s) if self.entity_anchor_matching == "v1" else em.normalize_strict(s)

    def _entity_matches(self, query_ent_norm: str, event_ent_norm: str) -> bool:
        from .delta_rag import entity_matching as em
        if self.entity_anchor_matching == "v1":
            return em.match_v1(query_ent_norm, event_ent_norm)
        return em.match_strict(query_ent_norm, event_ent_norm)

    async def _build_entity_index(self):
        from .delta_rag import entity_matching as em
        if self._entity_event_index is not None:
            return self._entity_event_index
        index = {}
        if not self.event_dynamic_graph:
            self._entity_event_index = index
            return index
        try:
            all_nodes = await self.event_dynamic_graph.get_all_nodes()
        except Exception as e:
            logger.warning(f"EATR: could not load nodes for entity index: {e}")
            all_nodes = None
        for nid, data in (all_nodes or {}).items():
            if not data:
                continue
            ents = data.get("entities_involved", [])
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except Exception:
                    ents = [ents]
            for e in (ents or []):
                key = self._normalize_entity(e)
                if key and (self.entity_anchor_matching == "v1" or em.valid_index_key(key)):
                    index.setdefault(key, []).append((nid, data))
        self._entity_event_index = index
        logger.info(f"EATR: built entity index over {len(index)} distinct entities")
        return index

    async def _build_conflict_index(self):
        if self._conflict_index is None:
            from .delta_rag.conflicts import ConflictIndex
            nodes = await self.event_dynamic_graph.get_all_nodes() if self.event_dynamic_graph else {}
            chunk_doc = {cid: (c or {}).get("full_doc_id", cid) for cid, c in self.text_chunks._data.items()}
            self._conflict_index = ConflictIndex(nodes or {}, chunk_doc)
            logger.info(f"Conflict index: {self._conflict_index.stats()}")
        return self._conflict_index

    async def _entity_anchored_events(self, query_entities, time_constraints, top_m, question=None):
        if not query_entities:
            return []
        index = await self._build_entity_index()
        if not index:
            return []
        if self.entity_anchor_matching != "v1" and question:
            from .delta_rag.entity_matching import filter_query_entities
            query_entities = filter_query_entities(query_entities, question)
        q_norms = [self._normalize_entity(e) for e in query_entities if self._normalize_entity(e)]

        seen, matched = set(), []
        for key, items in index.items():
            if any(self._entity_matches(qn, key) for qn in q_norms):
                for nid, data in items:
                    if nid not in seen:
                        seen.add(nid)
                        matched.append((nid, data))
        if not matched:
            return []

        from .delta_rag.temporal_index import day_index as to_days
        anchor = time_constraints.get("start_time") or time_constraints.get("end_time")
        anchor_days = to_days(anchor)

        def key_fn(item):
            d = to_days(item[1].get("timestamp"))
            if anchor_days is None or d is None:
                return (1, 0)
            return (0, abs(d - anchor_days))
        matched.sort(key=key_fn)

        results = []
        for nid, data in matched[:top_m]:
            ents = data.get("entities_involved", [])
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except Exception:
                    ents = [ents]
            results.append({
                "id": nid,
                "timestamp": data.get("timestamp", "static"),
                "sentence": data.get("sentence", ""),
                "context": data.get("context", ""),
                "entities_involved": ents,
                "source_id": data.get("source_id", ""),
                "distance": 0.0,
                "entity_anchored": True,
            })
        logger.info(f"EATR: anchored {len(results)} events for entities={query_entities}")
        return results

    def _init_arbiter(self):
        from .arbiter import ARBITER, ARBITERConfig
        from .arbiter.arbitration import ACAConfig
        from .arbiter.assertions import TANConfig
        from .arbiter.rendering import ACRConfig
        from .arbiter.timeline import STCConfig

        cfg = ARBITERConfig(
            tan=TANConfig(
                enable_interval_recovery=self.arbiter_enable_interval_recovery,
            ),
            stc=STCConfig(
                enable_supersession_closure=self.arbiter_enable_supersession_closure,
                enable_terminator_closure=self.arbiter_enable_terminator_closure,
            ),
            aca=ACAConfig(
                open_ended_penalty=self.arbiter_open_ended_penalty,
                min_holder_score=self.arbiter_min_holder_score,
                enable_corroboration=self.arbiter_enable_corroboration,
            ),
            acr=ACRConfig(
                max_slots=self.arbiter_max_slots,
                max_unslotted_events=self.arbiter_max_unslotted_events,
                retain_alternatives=self.arbiter_retain_alternatives,
                show_arbitration=self.arbiter_show_arbitration,
            ),
            require_conflict=self.arbiter_require_conflict,
        )
        self._arbiter = ARBITER(cfg)
        return self._arbiter

    def _run_arbiter(self, events, entities, time_constraints):
        if not self.enable_arbiter or not events:
            return None
        try:
            if self._arbiter is None:
                self._init_arbiter()
            block, stats = self._arbiter.arbitrate(events, entities, time_constraints)
            self._last_arbiter_stats = stats.as_dict()
            return block
        except Exception as e:
            logger.warning(f"ARBITER arbitration failed, falling back: {e}")
            return None

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "dynamic":
            response = await self.dynamic_query(
                query,
                param
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def dynamic_query(self, query: str, param: QueryParam):
        logger.info(f"Executing dynamic event query: {query}")
        
        time_constraints, entities = await self.parse_query_time_and_entities(
            query, param, self.llm_response_cache
        )
        logger.info(f"Extracted time constraints: {time_constraints}")
        logger.info(f"Extracted entities: {entities}")
        
        topk1 = param.topk1 or 2000  
        et_top_k = param.et_top_k or 20   
        found_results = []
        
        

        if hasattr(self.events_vdb, 'time_weighted_query'):

            start_time = time_constraints.get("start_time")
            end_time = time_constraints.get("end_time")
            current_timestamp = datetime.now().strftime("%Y-%m-%d")

            if not start_time:
                start_time = "0"

            if not end_time:
                end_time = current_timestamp

            all_time_weighted_results = []

            try:
                start_results = await self.events_vdb.time_weighted_query(
                    query=query, 
                    timestamp=start_time,
                    time_weight=getattr(self, 'time_weight', 1.0),
                    top_k=topk1 
                )
                if start_results:
                    all_time_weighted_results.extend(start_results)
                    logger.info(f"Found {len(start_results)} events using start_time query")
                else:
                    logger.info("No results from start_time query")
            except Exception as e:
                logger.warning(f"Error in time_weighted_query with start_time {start_time}: {e}")
            
            if end_time != start_time:
                try:
                    end_results = await self.events_vdb.time_weighted_query(
                        query=query,
                        timestamp=end_time,
                        time_weight=getattr(self, 'time_weight', 1.0),
                        top_k=topk1 
                    )
                    if end_results:
                        all_time_weighted_results.extend(end_results)
                        logger.info(f"Found {len(end_results)} events using end_time query")
                    else:
                        logger.info("No results from end_time query")
                except Exception as e:
                    logger.warning(f"Error in time_weighted_query with end_time {end_time}: {e}")
            else:
                logger.warning("start_time and end_time are the same, skipping duplicate query")
            
            if all_time_weighted_results:
                seen_ids = set()
                deduplicated_results = []
                for result in all_time_weighted_results:
                    event_id = result.get('id')
                    if event_id and event_id not in seen_ids:
                        seen_ids.add(event_id)
                        deduplicated_results.append(result)
                    elif not event_id: 
                        deduplicated_results.append(result)
                
                found_results = deduplicated_results
                logger.info(f"After deduplication: {len(found_results)} unique events")
            else:
                logger.warning("No events found using time_weighted_query")
                try:
                    fallback_results = await self.events_vdb.query(query, top_k=topk1)
                    if fallback_results:
                        found_results = fallback_results
                except Exception as e:
                    logger.warning(f"Error in fallback regular query: {e}")
                    
        else:
            regular_results = await self.events_vdb.query(query, top_k=topk1)
            if regular_results:
                found_results = regular_results
        
        top_k_seed_events = []
        if found_results:
            logger.info(f"Starting cross-encoder filtering to select top_{et_top_k} seed events from {len(found_results)} candidates")
            
            if (self.enable_ce_rerank or self.enable_bm25_reranking):
                if self.enable_ce_rerank and self.cross_encoder:
                    logger.info("Using cross-encoder for seed selection")
                    reranked_candidates = await self.rerank_with_cross_encoder(
                        events=found_results,
                        query=query,
                        entities=entities,
                        time_constraints=time_constraints
                    )
                elif self.enable_bm25_reranking:
                    logger.info("Using BM25 for seed selection")  
                    reranked_candidates = await self.rerank_with_bm25(
                        events=found_results,
                        query=query,
                        entities=entities,
                        time_constraints=time_constraints
                    )
                else:
                    reranked_candidates = found_results
                
                top_k_seed_events = reranked_candidates[:et_top_k]
                logger.info(f"Selected top_{et_top_k} events as seeds for graph traversal")
            else:
                sorted_candidates = sorted(found_results, key=lambda x: x.get('distance', 1.0))
                top_k_seed_events = sorted_candidates[:et_top_k]
                logger.info(f"No reranking enabled, selected top_{et_top_k} events by distance")
        else:
            logger.info("No filtered candidates available for seed selection")

        ahkr_results = []
        used_ahkr = False

        if (self.enable_delta_rag and self.enable_ahkr
                and self._delta_rag_ahkr is not None
                and self._delta_rag_tkps
                and self._delta_rag_embeddings is not None
                and self._delta_rag_events):

            logger.info("DELTA-RAG: Running AHKR for temporal-structural supplementation")
            try:
                if not self._delta_rag_tkps:
                    self._load_delta_rag_state()

                query_emb_batch = await self.embedding_func([query])
                query_embedding = query_emb_batch[0]

                if self._delta_rag_mstc is not None and self._delta_rag_mstc._trained:
                    import torch as _torch
                    with _torch.no_grad():
                        self._delta_rag_mstc.projection_head.eval()
                        q_tensor = _torch.from_numpy(query_embedding).float().unsqueeze(0).to(self._delta_rag_mstc.device)
                        query_embedding = self._delta_rag_mstc.projection_head(q_tensor).cpu().numpy()[0]

                ahkr_results = self._delta_rag_ahkr.retrieve(
                    query_embedding=query_embedding,
                    tkps=self._delta_rag_tkps,
                    events=self._delta_rag_events,
                    embeddings=self._delta_rag_embeddings,
                    time_constraints=time_constraints,
                    cpkb=self._delta_rag_cpkb,
                    entities=entities,
                    query_text=query,
                )
                if ahkr_results:
                    used_ahkr = True
                    logger.info(
                        f"DELTA-RAG AHKR: Retrieved {len(ahkr_results)} events "
                        f"(query_type={self._delta_rag_ahkr.last_query_type})"
                    )

            except Exception as e:
                logger.warning(f"DELTA-RAG AHKR failed: {e}")
                ahkr_results = []

        used_cwhr = False

        if (not used_ahkr
                and self.enable_cast_rag and self.enable_cwhr
                and self._cast_rag_cwhr is not None
                and self._cast_rag_tecs
                and self._cast_rag_embeddings is not None
                and self._cast_rag_events):

            logger.info("CAST-RAG: Using Confidence-Weighted Hierarchical Retrieval (CWHR)")
            try:
                if not self._cast_rag_tecs:
                    self._load_cast_rag_state()

                query_emb_batch = await self.embedding_func([query])
                query_embedding = query_emb_batch[0]

                if self._cast_rag_cte is not None and self._cast_rag_cte._trained:
                    import torch as _torch
                    with _torch.no_grad():
                        self._cast_rag_cte.projection_head.eval()
                        q_tensor = _torch.from_numpy(query_embedding).float().unsqueeze(0).to(self._cast_rag_cte.device)
                        query_embedding = self._cast_rag_cte.projection_head(q_tensor).cpu().numpy()[0]

                cwhr_results = self._cast_rag_cwhr.retrieve(
                    query_embedding=query_embedding,
                    tecs=self._cast_rag_tecs,
                    events=self._cast_rag_events,
                    embeddings=self._cast_rag_embeddings,
                    time_constraints=time_constraints,
                    bridging=self._cast_rag_seb,
                    entities=entities,
                )

                if cwhr_results:
                    final_results = cwhr_results
                    used_cwhr = True
                    logger.info(f"CAST-RAG CWHR: Retrieved {len(final_results)} events")

            except Exception as e:
                logger.warning(f"CAST-RAG CWHR failed, falling back to random walk: {e}")
                used_cwhr = False

        if used_cwhr:
            pass
        else:
            final_results = []
            logger.info(f"Starting graph traversal from {len(top_k_seed_events)} seed events")

            seed_event_ids = []
            for result in top_k_seed_events:
                event_id = result.get('id')
                if event_id:
                    seed_event_ids.append(event_id)

            graph_traversed_event_ids = []
            if (self.enable_graph_traversal and seed_event_ids and self.event_dynamic_graph):
                try:
                    graph_traversed_event_ids = await self._random_walk_graph_traversal(
                        event_graph=self.event_dynamic_graph,
                        seed_event_ids=seed_event_ids,
                        max_depth=self.walk_depth,
                        max_nodes_per_seed=self.walk_nodes,
                        num_walks=self.walk_n
                    )
                    logger.info(f"Graph traversal found {len(graph_traversed_event_ids)} total events")
                except Exception as e:
                    logger.info(f"Error during graph traversal: {e}")
                    graph_traversed_event_ids = []
            elif not self.enable_graph_traversal:
                logger.info("Graph traversal is disabled in configuration")
            else:
                logger.info("Skipping graph traversal: no valid seeds or graph not available")

            if graph_traversed_event_ids:
                if self.event_dynamic_graph:
                    try:
                        events_data = await self.event_dynamic_graph.get_nodes_batch(graph_traversed_event_ids)
                        for i, event_data in enumerate(events_data):
                            if event_data is not None:
                                event_id = graph_traversed_event_ids[i]
                                final_results.append(
                                    {
                                        "id": event_id,
                                        "timestamp": event_data.get("timestamp", "unknown"),
                                        "sentence": event_data.get("sentence", ""),
                                        "context": event_data.get("context", ""),
                                        "entities_involved": event_data.get("entities_involved", []),
                                        "source_id": event_data.get("source_id", ""),
                                        "distance": 0.0,
                                    }
                                )
                        logger.info(f"Retrieved {len(final_results)} complete events from event_dynamic_graph")
                    except Exception as e:
                        logger.error(f"Error retrieving events from event_dynamic_graph: {e}")
                        final_results = []
                else:
                    logger.warning("event_dynamic_graph not available")
                    final_results = []
            else:
                logger.info("No events found through graph traversal")
        
        if used_ahkr and ahkr_results:
            merge_budget = int(et_top_k * 1.5)
            seen_ids = set()
            merged = []
            for ev in top_k_seed_events:
                eid = ev.get('id', '')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    merged.append(ev)
            for ev in final_results:
                eid = ev.get('id', '')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    merged.append(ev)
            for ev in ahkr_results:
                eid = ev.get('id', '')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    merged.append(ev)
            final_results = merged[:merge_budget]
            logger.info(
                f"DELTA-RAG three-way merge: {len(top_k_seed_events)} CE + "
                f"{len(final_results)} graph + {len(ahkr_results)} AHKR "
                f"= {len(final_results)} events (budget={merge_budget})"
            )

        if self.enable_entity_anchor and entities:
            try:
                anchored = await self._entity_anchored_events(
                    entities, time_constraints, self.entity_anchor_top_m, question=query
                )
                if anchored:
                    existing_ids = {ev.get('id', '') for ev in final_results}
                    new_anchored = [ev for ev in anchored if ev.get('id', '') not in existing_ids]
                    final_results = new_anchored + final_results
                    logger.info(
                        f"EATR: injected {len(new_anchored)} anchored events "
                        f"(final_results now {len(final_results)})"
                    )
            except Exception as e:
                logger.warning(f"EATR injection failed: {e}")

        if (self.event_hook is not None or self.enable_conflict_fusion) and final_results and self.event_dynamic_graph:
            need = [i for i, ev in enumerate(final_results)
                    if ev.get("id") and (not ev.get("entities_involved") or not ev.get("source_id"))]
            if need:
                gdata = await self.event_dynamic_graph.get_nodes_batch([final_results[i]["id"] for i in need])
                for i, g in zip(need, gdata):
                    if g:
                        final_results[i] = {**final_results[i],
                                            "entities_involved": final_results[i].get("entities_involved") or g.get("entities_involved", []),
                                            "source_id": final_results[i].get("source_id") or g.get("source_id", "")}
        if self.event_hook is not None:
            final_results = self.event_hook(final_results, query)
        self.last_conflict_stats = {}
        if self.enable_conflict_fusion and final_results:
            from .delta_rag.conflicts import fuse
            idx = await self._build_conflict_index()
            final_results, self.last_conflict_stats = fuse(final_results, idx, self.source_credibility)

        arbitrated_context = self._run_arbiter(
            final_results, entities, time_constraints
        )

        if getattr(param, "diagnostic_only", False):
            return {
                "n_events": len(final_results),
                "entities": entities,
                "time_constraints": time_constraints,
                "arbiter_applied": bool(arbitrated_context),
                "arbiter_stats": self._last_arbiter_stats,
            }

        if final_results:
            logger.info(f"Using {len(final_results)} events for context building")

        if final_results and self.event_dynamic_graph:
            missing_source_ids = [
                (i, ev.get('id', '')) for i, ev in enumerate(final_results)
                if not ev.get('source_id') and ev.get('id')
            ]
            if missing_source_ids:
                logger.info(f"Enriching {len(missing_source_ids)} events missing source_id from graph")
                try:
                    ids_to_fetch = [eid for _, eid in missing_source_ids]
                    graph_data = await self.event_dynamic_graph.get_nodes_batch(ids_to_fetch)
                    enriched = 0
                    for (idx, eid), gdata in zip(missing_source_ids, graph_data):
                        if gdata and gdata.get('source_id'):
                            final_results[idx]['source_id'] = gdata['source_id']
                            if not final_results[idx].get('context') and gdata.get('context'):
                                final_results[idx]['context'] = gdata['context']
                            enriched += 1
                    if enriched:
                        logger.info(f"Enriched {enriched}/{len(missing_source_ids)} events with source_id from graph")
                except Exception as e:
                    logger.warning(f"Failed to enrich events from graph: {e}")

        sorted_final_results = sorted(final_results, key=lambda x: x.get('timestamp', '0'))

        question = query

        from .prompt import PROMPTS, GRAPH_FIELD_SEP
        from collections import Counter

        source_id_frequency = Counter()

        for i, event in enumerate(final_results):
            event_source_id = event.get('source_id', '')
            logger.debug(f"Event {i+1}: ID={event.get('id', 'N/A')}, source_id='{event_source_id}'")
            
            if event_source_id:
                event_source_ids = event_source_id.split(GRAPH_FIELD_SEP)
                for sid in event_source_ids:
                    sid = sid.strip()
                    if sid:
                        source_id_frequency[sid] += 1
        
        if source_id_frequency:
            sorted_source_ids = [sid for sid, freq in source_id_frequency.most_common()]
            limited_source_ids = sorted_source_ids[:param.top_k]
            
            logger.info(f"Found {len(source_id_frequency)} unique source_ids")
        else:
            limited_source_ids = []
            logger.warning("No valid source_ids found in events")
        
        try:
            retrieved_chunks = await self.text_chunks.get_by_ids(limited_source_ids)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from text_chunks")

            for i, chunk in enumerate(retrieved_chunks):
                if chunk is not None:
                    chunk_keys = list(chunk.keys()) if isinstance(chunk, dict) else "Not a dict"
                    has_content = 'content' in chunk if isinstance(chunk, dict) else False
                    content_length = len(chunk.get('content', '')) if isinstance(chunk, dict) and chunk.get('content') else 0
                    logger.debug(f"Chunk {i+1}: keys={chunk_keys}, has_content={has_content}, content_length={content_length}")
            
            valid_chunks = []
            for chunk in retrieved_chunks:
                if chunk is not None and isinstance(chunk, dict):
                    content = chunk.get('content', '').strip()
                    if content:
                        valid_chunks.append(content)
            
            logger.debug(f"Extracted {len(valid_chunks)} valid chunks with content")
            
            truncated_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x,
                max_token_size=param.max_token_for_text_unit
            )
            logger.debug(f"After token truncation: {len(truncated_chunks)} chunks remain")
            
            events_section = ""
            if final_results and self.if_timeline_events:
                if arbitrated_context:
                    events_section = arbitrated_context
                    logger.debug("Using ARBITER arbitrated state timeline as events section")
                else:
                    events_section = self.build_time_CoT(
                        events=final_results,
                        query=question,
                        time_constraints=time_constraints,
                        entities=entities
                    )
            elif not self.if_timeline_events:
                logger.info("Timeline events disabled for ablation study - skipping events section construction")
            
            chunks_section = ""
            if truncated_chunks:
                for i, chunk_content in enumerate(truncated_chunks):
                    chunks_section += f"## Chunk {i+1}\n{chunk_content}\n\n"
            else:
                chunks_section = "No relevant chunks found."
            if self.if_timeline_events:
                context = PROMPTS["dynamic_QA"].format(
                    question=question,
                    events_data=events_section,
                    chunks_data=chunks_section
                )
                logger.debug("Using dynamic_QA prompt template with events and chunks")
            else:
                context = PROMPTS["dynamic_QA_wo_timeline"].format(
                    question=question,
                    chunks_data=chunks_section
                )
                logger.debug("Using dynamic_QA_wo_timeline prompt template with chunks only (ablation study)")
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            events_section = ""
            if final_results:
                events_section = arbitrated_context or self.build_time_CoT(
                    events=final_results,
                    query=question,
                    time_constraints=time_constraints,
                    entities=entities
                )

            context = PROMPTS["short_answer"].format(
                content_data=events_section, 
                response_type=param.response_type
            )
        
        if self.answer_commit:
            context += (
                "\n\n# FINAL ANSWER (required)\n"
                "After your analysis, end with exactly one line of the form\n"
                "Final answer: <answer>\n"
                "Commit to the single answer that the evidence supports for the asked time "
                "(give several items only if the question asks for several). Do not list "
                "alternative candidates in this line. If the evidence does not contain the "
                "answer, write: Final answer: not available\n"
            )
        response = await self.best_model_func(context)
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                await self._insert_done()
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                await self._insert_done()
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            logger.info("[Event Extraction]...")
            maybe_new_dyg, extraction_stats = await extract_events(
                inserting_chunks,
                dyg_inst=self.event_dynamic_graph,
                events_vdb = self.events_vdb,
                global_config={**self.get_config_dict(), "events_vdb": self.events_vdb},
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            self.extraction_stats = extraction_stats
            
            if maybe_new_dyg is not None:
                self.event_dynamic_graph = maybe_new_dyg
                logger.info(f"Updated dynamic event graph")
            else:
                logger.warning("No new events found")
                await self._insert_done()
                return
            
            torch.cuda.empty_cache()
            time.sleep(2)

            if self.enable_cast_rag:
                try:
                    all_events_for_cast = []
                    all_nodes = await self.event_dynamic_graph.get_all_nodes()
                    if all_nodes:
                        for node_id in all_nodes:
                            node_data = await self.event_dynamic_graph.get_node(node_id)
                            if node_data:
                                node_data["event_id"] = node_id
                                all_events_for_cast.append(node_data)

                    if all_events_for_cast:
                        await self._run_cast_rag_pipeline(all_events_for_cast)
                    else:
                        logger.warning("CAST-RAG: No events available for enhancement pipeline")
                except Exception as e:
                    logger.error(f"CAST-RAG pipeline error (non-fatal): {e}")

            if self.enable_delta_rag:
                try:
                    all_events_for_delta = []
                    all_nodes = await self.event_dynamic_graph.get_all_nodes()
                    if all_nodes:
                        for node_id in all_nodes:
                            node_data = await self.event_dynamic_graph.get_node(node_id)
                            if node_data:
                                node_data["event_id"] = node_id
                                all_events_for_delta.append(node_data)

                    if all_events_for_delta:
                        await self._run_delta_rag_pipeline(all_events_for_delta)
                    else:
                        logger.warning("DELTA-RAG: No events available for enhancement pipeline")
                except Exception as e:
                    logger.error(f"DELTA-RAG pipeline error (non-fatal): {e}")

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.events_vdb,
            self.event_dynamic_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.event_dynamic_graph,
            self.events_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def build_time_CoT(self, events, query=None, time_constraints=None, entities=None):
        if not events:
            return ""

        static_events = []
        timestamped_events = []
        
        for event in events:
            timestamp = event.get('timestamp', 'static')
            if timestamp == 'static' or not timestamp or timestamp == 'unknown':
                static_events.append(event)
            else:
                timestamped_events.append(event)
        
        timestamped_events.sort(key=lambda x: x.get('timestamp', '0'))
        
        sorted_events = static_events + timestamped_events

        context_parts = []

        for i, event in enumerate(sorted_events):
            event_time = event.get('timestamp', 'static')
            sentence = event.get('sentence', '')
            context = event.get('context', '')
            
            if event_time == 'static' or not event_time or event_time == 'unknown':
                time_display = 'Static'
            else:
                time_display = event_time

            event_text = f"Event #{i+1} [{time_display}]: {sentence}"
            context_parts.append(event_text)

        return "\n".join(context_parts)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["cheap_model_func", "best_model_func", "cross_encoder"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    async def _random_walk_graph_traversal(
        self,
        event_graph,
        seed_event_ids,
        max_depth=3,
        max_nodes_per_seed=10,
        num_walks=5,
    ):
        if not seed_event_ids:
            return []

        rng = random.Random(
            f"{self.random_seed}|{'|'.join(sorted(seed_event_ids))}"
        )

        all_nodes = await event_graph.get_all_nodes()
        if not all_nodes:
            logger.info("Graph is empty")
            return []

        valid_seeds = [
            seed_id
            for seed_id in seed_event_ids
            if await event_graph.has_node(seed_id)
        ]
        if not valid_seeds:
            logger.info("No valid seed events found in the graph")
            return []

        logger.info(
            f"Starting random walk traversal from {len(valid_seeds)} seed events"
        )

        all_related_events = set()

        for seed_event_id in valid_seeds:
            seed_related_events = set()
            seed_related_events.add(seed_event_id)

            for walk_num in range(num_walks):
                current_node = seed_event_id
                walk_path = [current_node]

                for step in range(max_depth):
                    neighbors = []
                    weights = []
                    node_edges = await event_graph.get_node_edges(current_node)

                    if node_edges:
                        for edge in node_edges:
                            neighbor_node = (
                                edge[1] if edge[0] == current_node else edge[0]
                            )

                            edge_data = await event_graph.get_edge(edge[0], edge[1])
                            if (
                                edge_data
                                and edge_data.get("relation_type")
                                in ("event_temporal_proximity", "semantic_bridge")
                            ):
                                neighbors.append(neighbor_node)
                                weights.append(edge_data.get("weight", 1.0))

                    available_indices = [
                        i for i, n in enumerate(neighbors) if n not in walk_path
                    ]
                    if not available_indices:
                        break
                    
                    available_weights = [weights[i] for i in available_indices]
                    available_neighbors = [neighbors[i] for i in available_indices]
                    
                    total_weight = sum(available_weights)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in available_weights]
                        current_node = rng.choices(
                            available_neighbors, weights=probabilities, k=1
                        )[0]
                    else:
                        current_node = rng.choice(available_neighbors)
                    
                    walk_path.append(current_node)
                    seed_related_events.add(current_node)
                    
                    if len(seed_related_events) >= max_nodes_per_seed:
                        break

                await asyncio.sleep(0)

                if len(seed_related_events) >= max_nodes_per_seed:
                    break

            all_related_events.update(seed_related_events)
            logger.info(
                f"Seed {seed_event_id}: collected {len(seed_related_events)} related events"
            )

        related_event_list = list(all_related_events)
        logger.info(
            f"Random walk traversal completed: {len(related_event_list)} total unique events found"
        )

        return related_event_list

    def tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        import string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [token for token in text.split() if token.strip()]
        return tokens

    def build_bm25_corpus(self, events: List[Dict]) -> tuple[List[List[str]], Dict[str, int]]:
        corpus = []
        event_id_to_index = {}
        
        for idx, event in enumerate(events):
            text_content = ""
            
            if event.get('sentence'):
                text_content = event.get('sentence', '')
            elif event.get('content'):
                text_content = event.get('content', '')
            else:
                text_content = ""
            
            tokens = self.tokenize_text(text_content.strip())
            corpus.append(tokens)
            
            event_id = event.get('id')
            if event_id:
                event_id_to_index[event_id] = idx
        
        return corpus, event_id_to_index

    def calculate_bm25_scores(self, query: str, corpus: List[List[str]], 
                            k1: float = 1.5, b: float = 0.75) -> List[float]:
        if not corpus:
            return []
        
        query_tokens = self.tokenize_text(query)
        if not query_tokens:
            return [0.0] * len(corpus)
        
        N = len(corpus)
        doc_freqs = defaultdict(int)
        doc_lengths = []
        
        for doc in corpus:
            doc_lengths.append(len(doc))
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        scores = []
        for doc_idx, doc in enumerate(corpus):
            score = 0.0
            doc_length = doc_lengths[doc_idx]
            
            term_freqs = Counter(doc)
            
            for term in query_tokens:
                if term in term_freqs:
                    tf = term_freqs[term]
                    
                    df = doc_freqs[term]
                    
                    idf = math.log((N - df + 0.5) / (df + 0.5))
                    
                    numerator = idf * tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                    
                    score += numerator / denominator
            
            scores.append(score)
        
        return scores

    def calculate_entity_match_score(self, event_content: str, entities: List[str]) -> float:
        if not entities or not event_content:
            return 0.0
        
        content_lower = event_content.lower()
        matched_entities = 0
        
        for entity in entities:
            if self._is_entity_matched(entity, content_lower):
                matched_entities += 1
        
        return matched_entities / len(entities)
    
    def _is_entity_matched(self, entity: str, content_lower: str) -> bool:
        entity_lower = entity.lower().strip()
        
        if entity_lower in content_lower:
            return True
        
        entity_words = [word.strip() for word in entity_lower.split() if len(word.strip()) > 2]
        
        if len(entity_words) > 1:
            for word in entity_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, content_lower):
                    return True
        
        if ' ' in entity_lower:
            last_name = entity_words[-1] if entity_words else ""
            if len(last_name) > 2:
                pattern = r'\b' + re.escape(last_name) + r'\b'
                if re.search(pattern, content_lower):
                    return True
        
        entity_initials = ''.join([word[0] for word in entity_words if word])
        if len(entity_initials) > 1:
            pattern = r'\b' + re.escape(entity_initials) + r'\b'
            if re.search(pattern, content_lower):
                return True
        
        if hasattr(self, 'enable_fuzzy_entity_matching') and self.enable_fuzzy_entity_matching:
            threshold = getattr(self, 'fuzzy_match_threshold', 0.8)
            return self._fuzzy_entity_match(entity_lower, content_lower, threshold)
        
        return False
    
    def _fuzzy_entity_match(self, entity: str, content: str, threshold: float = 0.8) -> bool:
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return False
        
        content_words = content.split()
        
        for word in content_words:
            if len(word) > 2:
                similarity = SequenceMatcher(None, entity.lower(), word.lower()).ratio()
                if similarity >= threshold:
                    return True
        
        entity_words = entity.split()
        if len(entity_words) > 1:
            content_words = content.split()
            for i in range(len(content_words) - len(entity_words) + 1):
                content_phrase = ' '.join(content_words[i:i+len(entity_words)])
                similarity = SequenceMatcher(None, entity.lower(), content_phrase.lower()).ratio()
                if similarity >= threshold:
                    return True
        
        return False

    async def rerank_with_bm25(self, events: List[Dict], query: str, 
                             entities: List[str], time_constraints: Dict) -> List[Dict]:
        if not events or not self.enable_bm25_reranking:
            return events
        
        logger.info(f"Starting BM25 reranking for {len(events)} events")
        
        corpus, event_id_to_index = self.build_bm25_corpus(events)
        
        bm25_scores = self.calculate_bm25_scores(
            query=query, 
            corpus=corpus, 
            k1=self.bm25_k1, 
            b=self.bm25_b
        )
        
        if bm25_scores and max(bm25_scores) > 0:
            max_score = max(bm25_scores)
            bm25_scores = [score / max_score for score in bm25_scores]
        
        scored_events = []
        for idx, event in enumerate(events):
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            
            entity_score = self.calculate_entity_match_score(
                event.get('sentence', '') or event.get('content', ''), entities
            )
            
            composite_score = (
                bm25_score * self.bm25_weight +
                entity_score * self.entity_match_weight
            )
            
            event['_bm25_score'] = bm25_score
            event['_entity_score'] = entity_score
            event['_composite_score'] = composite_score
            
            scored_events.append(event)
        
        reranked_events = sorted(scored_events, 
                               key=lambda x: x['_composite_score'], 
                               reverse=True)
        
        logger.info(f"BM25 reranking completed")
        
        return reranked_events

    async def rerank_with_cross_encoder(self, events: List[Dict], query: str, 
                                      entities: List[str], time_constraints: Dict) -> List[Dict]:
        if not events or not self.enable_ce_rerank or not self.cross_encoder:
            return events
        
        logger.info(f"Starting cross-encoder reranking for {len(events)} events")
        
        return await self._rerank_with_cross_encoder(events, query, entities, time_constraints)
        

    
    async def _rerank_with_cross_encoder(self, events: List[Dict], query: str, 
                                                 entities: List[str], time_constraints: Dict) -> List[Dict]:
          
        query_doc_pairs = []
        cache_keys = []
        cached_scores = []
        uncached_indices = []
        
        truncated_query = (
            query[: self.ce_truncate_len]
            if len(query) > self.ce_truncate_len
            else query
        )

        for idx, event in enumerate(events):
            event_text = event.get('sentence', '') or event.get('content', '')
            
            if event_text and len(event_text) > self.ce_truncate_len:
                event_text = event_text[:self.ce_truncate_len]
            
            cache_key = f"{hash(truncated_query)}_{hash(event_text)}"
            cache_keys.append(cache_key)
            
            if hasattr(self, 'cross_encoder_cache') and cache_key in self.cross_encoder_cache:
                cached_scores.append(self.cross_encoder_cache[cache_key])
            else:
                cached_scores.append(None)
                uncached_indices.append(idx)
                if event_text:
                    query_doc_pairs.append([truncated_query, event_text.strip()])
                else:
                    query_doc_pairs.append([truncated_query, ""])

        logger.info(
            f"Cache hit rate: {(len(events) - len(uncached_indices)) / len(events) * 100:.1f}% ({len(events) - len(uncached_indices)}/{len(events)})"
        )

        cross_encoder_scores = [0.0] * len(events)
        
        if query_doc_pairs:
            try:
                batch_size = self._get_optimal_batch_size(len(query_doc_pairs))
                logger.info(f"Using dynamic batch size: {batch_size}")
                
                uncached_scores = []
                for i in range(0, len(query_doc_pairs), batch_size):
                    batch_pairs = query_doc_pairs[i:i + batch_size]
                    batch_scores = self.cross_encoder.predict(batch_pairs)
                    
                    if hasattr(batch_scores, 'tolist'):
                        batch_scores = batch_scores.tolist()
                    elif not isinstance(batch_scores, list):
                        batch_scores = [float(score) for score in batch_scores]
                    
                    uncached_scores.extend(batch_scores)
                    
                    await asyncio.sleep(0)
                
                uncached_idx = 0
                for idx in range(len(events)):
                    if cached_scores[idx] is not None:
                        cross_encoder_scores[idx] = cached_scores[idx]
                    else:
                        score = uncached_scores[uncached_idx] if uncached_idx < len(uncached_scores) else 0.0
                        cross_encoder_scores[idx] = score
                        
                        if hasattr(self, 'cross_encoder_cache'):
                            if len(self.cross_encoder_cache) >= self.ce_cache_size:
                                oldest_key = next(iter(self.cross_encoder_cache))
                                del self.cross_encoder_cache[oldest_key]
                            self.cross_encoder_cache[cache_keys[idx]] = score

                        uncached_idx += 1
                        
            except Exception as e:
                logger.error(f"Error during cross-encoder prediction: {e}")
                cross_encoder_scores = [0.5] * len(events)
        else:
            for idx in range(len(events)):
                if cached_scores[idx] is not None:
                    cross_encoder_scores[idx] = cached_scores[idx]
        
        if cross_encoder_scores and len(cross_encoder_scores) > 0:
            min_score = min(cross_encoder_scores)
            max_score = max(cross_encoder_scores)
            if max_score > min_score:
                cross_encoder_scores = [
                    (score - min_score) / (max_score - min_score) 
                    for score in cross_encoder_scores
                ]
            else:
                cross_encoder_scores = [0.5] * len(cross_encoder_scores)
        
        if cross_encoder_scores and self.ce_early_stop > 0:
            max_score = max(cross_encoder_scores)
            if max_score >= self.ce_early_stop:
                logger.info(f"Early stopping triggered: top score {max_score:.3f} >= {self.ce_early_stop}")
        
        scored_events = []
        for idx, event in enumerate(events):
            cross_encoder_score = cross_encoder_scores[idx] if idx < len(cross_encoder_scores) else 0.0
            
            entity_score = self.calculate_entity_match_score(
                event.get('sentence', '') or event.get('content', ''), entities
            )
            
            composite_score = (
                cross_encoder_score * self.ce_weight +
                entity_score * self.ce_ent_weight
            )
            
            event['_cross_encoder_score'] = cross_encoder_score
            event['_entity_score'] = entity_score
            event['_composite_score'] = composite_score
            
            scored_events.append(event)
        
        reranked_events = sorted(scored_events, 
                               key=lambda x: x['_composite_score'], 
                               reverse=True)
        
        logger.info(f"Optimized cross-encoder reranking completed")
        
        return reranked_events

    def _get_optimal_batch_size(self, num_pairs: int) -> int:
        if not self.ce_dynamic_batch:
            return self.ce_batch_size
        
        optimal_size = self.ce_batch_size
        
        if num_pairs < 32:
            optimal_size = min(num_pairs, self.ce_min_batch)
        elif num_pairs > 200:
            optimal_size = min(self.ce_max_batch, optimal_size * 2)
        
        try:
            import torch
            if torch.cuda.is_available() and self.ce_device != "cpu":
                current_memory = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage_ratio = current_memory / total_memory
                
                if memory_usage_ratio > 0.8:
                    optimal_size = max(self.ce_min_batch, optimal_size // 2)
                elif memory_usage_ratio < 0.4:
                    optimal_size = min(self.ce_max_batch, optimal_size * 1.5)
        except:
            pass
        
        return max(self.ce_min_batch, min(self.ce_max_batch, int(optimal_size)))
    
    async def parse_query_time_and_entities(
        self,
        query: str,
        query_param: QueryParam,
        hashing_kv: BaseKVStorage | None = None,
    ) -> tuple[dict, list[str]]:

        if hashing_kv is None:
            hashing_kv = self.llm_response_cache
            
        if hasattr(query_param, 'time_constraints') and hasattr(query_param, 'entities'):
            if query_param.time_constraints and query_param.entities:
                return query_param.time_constraints, query_param.entities
            
        global_config = self.get_config_dict()
        time_constraints, entities = await self.aextract_time_and_entities(
            query, query_param, global_config, hashing_kv
        )
        return time_constraints, entities

    async def aextract_time_and_entities(
        self,
        query: str,
        param: QueryParam,
        global_config: dict,
        hashing_kv: BaseKVStorage | None = None,
    ) -> tuple[dict, list[str]]:

        args_hash = compute_args_hash(param.mode, query, "time_entities")
        cached_response = None
        quantized = False
        min_val = 0
        max_val = 0
        
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            try:
                cached_item = await hashing_kv.get_by_id(args_hash)
                if cached_item and (cached_item.get("cache_type") == "time_entities"):
                    cached_response = cached_item.get("content")
                    quantized = cached_item.get("quantized", False)
                    min_val = cached_item.get("min_val", 0)
                    max_val = cached_item.get("max_val", 0)
                    logger.info(f"Using cached time and entities data for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Error retrieving from cache: {e}")
        
        if cached_response:
            try:
                parsed_data = json.loads(cached_response)
                return parsed_data["time_constraints"], parsed_data["entities"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Invalid cache format for time constraints and entities: {e}, proceeding with extraction"
                )
        
        example_number = global_config.get("addon_params", {}).get("example_number", None)
        examples = ""
        if "time_entity_extraction_examples" in PROMPTS:
            if example_number and example_number < len(PROMPTS["time_entity_extraction_examples"]):
                examples = "\n".join(
                    PROMPTS["time_entity_extraction_examples"][: int(example_number)]
                )
            else:
                examples = "\n".join(PROMPTS["time_entity_extraction_examples"])
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        te_prompt = PROMPTS["time_entity_extraction"]
        te_prompt = te_prompt.replace("{query}", query)
        te_prompt = te_prompt.replace("{examples}", examples)
        te_prompt = te_prompt.replace("{current_date}", current_date)
        
        use_model_func = global_config.get("best_model_func")
        
        llm_kwargs = global_config.get("special_community_report_llm_kwargs", {})
        if not llm_kwargs:
            llm_kwargs = {"response_format": {"type": "json_object"}}
        
        result = await use_model_func(te_prompt, **llm_kwargs)
        
        try:
            parsed_data = json.loads(result)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if not match:
                logger.error("No JSON-like structure found in the LLM response.")
                return {"start_time": None, "end_time": None}, []
            try:
                parsed_data = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return {"start_time": None, "end_time": None}, []
        
        time_constraints = parsed_data.get("time_constraints", {"start_time": None, "end_time": None})
        entities = parsed_data.get("entities", [])
        
        if time_constraints.get("start_time"):
            time_constraints["start_time"] = normalize_timestamp(time_constraints["start_time"])
        if time_constraints.get("end_time"):
            time_constraints["end_time"] = normalize_timestamp(time_constraints["end_time"])
        

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            cache_data = {
                "time_constraints": time_constraints,
                "entities": entities,
            }
            await hashing_kv.upsert({
                args_hash: {
                    "content": json.dumps(cache_data),
                    "prompt": query,
                    "quantized": quantized,
                    "min_val": min_val,
                    "max_val": max_val,
                    "mode": param.mode,
                    "cache_type": "time_entities",
                }
            })
        
        return time_constraints, entities
