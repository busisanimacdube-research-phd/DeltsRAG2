import numpy as np
import math
import datetime
from dataclasses import dataclass
from dateutil import parser as date_parser
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import calendar

from .vdb_nanovectordb import NanoVectorDBStorage, NanoVectorDB
from .._utils import logger


def encode_timestamp(timestamp: str, dim: int = 8) -> np.ndarray:
    if timestamp == "static" or not timestamp:
        return np.zeros(dim)
    
    if dim % 4 != 0:
        raise ValueError("Timestamp encoding dimension must be a multiple of 4")
    
    year, month, day = 0, 0, 0
    precision_level = -1
    
    try:
        if len(timestamp) == 4:
            year = int(timestamp)
            month, day = 1, 1
            precision_level = 0
        elif len(timestamp) == 7:
            year_str, month_str = timestamp.split('-')
            year, month = int(year_str), int(month_str)
            if not (1 <= month <= 12):
                raise ValueError("Month out of range.")
            day = 1
            precision_level = 1
        elif len(timestamp) == 10:
            year_str, month_str, day_str = timestamp.split('-')
            year, month, day = int(year_str), int(month_str), int(day_str)
            datetime.datetime(year, month, day)
            precision_level = 2
        else:
            dt = date_parser.parse(timestamp)
            year, month, day = dt.year, dt.month, dt.day
            precision_level = 2
    except (ValueError, TypeError, AttributeError):
        return np.zeros(dim)
    
    t_val: float
    if precision_level == 0:
        t_val = float(year)
    elif precision_level == 1:
        t_val = float(year) + (float(month) - 0.5) / 12.0
    elif precision_level == 2:
        day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday
        days_in_year = 366.0 if calendar.isleap(year) else 365.0
        t_val = float(year) + (float(day_of_year) - 0.5) / days_in_year
    else:
        return np.zeros(dim)
    
    t_norm = (t_val - 1900.0) / 200.0
    
    encoding = np.zeros(dim)
    num_freq_bands = dim // 2
    
    for k in range(num_freq_bands):
        freq_val = (2.0 ** k) * math.pi * t_norm
        encoding[2 * k] = math.sin(freq_val)
        encoding[2 * k + 1] = math.cos(freq_val)
        
    return encoding


@dataclass
class TimestampEnhancedVectorStorage(NanoVectorDBStorage):
    timestamp_dim: int = 16
    
    def __post_init__(self):
        if self.timestamp_dim % 4 != 0:
            self.timestamp_dim = (self.timestamp_dim // 4) * 4
            logger.warning(f"Adjusted timestamp_dim to {self.timestamp_dim} to ensure it's a multiple of 4")
            
        embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
        self.original_dim = embed_dim
        self.enhanced_dim = self.original_dim + self.timestamp_dim
        
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        try:
            logger.info(f"Initializing TimestampEnhancedVectorDB with embedding dimension {self.enhanced_dim}")
            
            self._client = NanoVectorDB(
                self.enhanced_dim, storage_file=self._client_file_name
            )
            
            if hasattr(self._client, '__storage') and 'matrix' in self._client.__storage:
                if self._client.__storage['matrix'] is None or len(self._client.__storage['matrix']) == 0:
                    logger.warning("Vector database matrix is empty, it will be initialized on first upsert")
            
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
        except Exception as e:
            logger.error(f"Error initializing TimestampEnhancedVectorStorage: {e}")
            self._client = NanoVectorDB(
                self.enhanced_dim, storage_file=self._client_file_name
            )
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
            
        logger.info(f"Initialized TimestampEnhancedVectorStorage with original_dim={self.original_dim}, "
                   f"timestamp_dim={self.timestamp_dim}, enhanced_dim={self.enhanced_dim}")
    
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> bool:
        logger.info(f"Inserting {len(data)} vectors with timestamp enhancement to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
            
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        
        logger.debug(f"Processing {len(batches)} embedding batches with size up to {self._max_batch_size}")
        embeddings_list = []
        for batch in batches:
            emb = await self.embedding_func(batch)
            embeddings_list.append(emb)
            try:
                import torch, gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except ImportError:
                pass
        
        embeddings = np.concatenate(embeddings_list)
        
        timestamp_embeddings = []
        for doc_id, doc in data.items():
            timestamp = doc.get("timestamp", "static")
            ts_encoding = encode_timestamp(timestamp, self.timestamp_dim)
            timestamp_embeddings.append(ts_encoding)
        
        timestamp_embeddings = np.array(timestamp_embeddings)
        enhanced_embeddings = np.hstack((embeddings, timestamp_embeddings))
        
        for i, d in enumerate(list_data):
            d["__vector__"] = enhanced_embeddings[i]
            
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str, top_k: int = 5, time_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = await self.embedding_func([query])
        query_embedding = query_embedding[0]
        
        if time_filter:
            ts_encoding = encode_timestamp(time_filter, self.timestamp_dim)
        else:
            ts_encoding = np.zeros(self.timestamp_dim)
        
        enhanced_query = np.hstack((query_embedding, ts_encoding))
        
        results = self._client.query(
            query=enhanced_query,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def time_weighted_query(self, query: str, timestamp: str, time_weight: Optional[float] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = await self.embedding_func([query])
        query_embedding = query_embedding[0]
        
        ts_encoding = encode_timestamp(timestamp, self.timestamp_dim)
        
        actual_time_weight = time_weight if time_weight is not None else 1.0
        scaled_ts_encoding = ts_encoding * actual_time_weight
        
        enhanced_query = np.hstack((query_embedding, scaled_ts_encoding))
        
        results = self._client.query(
            query=enhanced_query,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} 
            for dp in results
        ]
        return results 