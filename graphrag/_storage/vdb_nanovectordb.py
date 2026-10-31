import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from typing import List, Optional

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        try:
            embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
            logger.info(f"Initializing NanoVectorDB with embedding dimension {embed_dim}")
            
            self._client = NanoVectorDB(
                embed_dim, storage_file=self._client_file_name
            )
            
            if hasattr(self._client, '__storage') and 'matrix' in self._client.__storage:
                if self._client.__storage['matrix'] is None or len(self._client.__storage['matrix']) == 0:
                    logger.warning("Vector database matrix is empty, it will be initialized on first upsert")
            
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
        except Exception as e:
            logger.error(f"Error initializing NanoVectorDBStorage: {e}")
            self._client = NanoVectorDB(
                1536, storage_file=self._client_file_name
            )
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
            
        try:
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
            
            logger.debug(f"Processing {len(batches)} embedding batches of size up to {self._max_batch_size} sequentially")
            embeddings_list = []
            for batch in batches:
                try:
                    emb = await self.embedding_func(batch)
                    if emb is None or len(emb) == 0:
                        logger.error(f"Embedding function returned empty result for batch of size {len(batch)}")
                        continue
                    embeddings_list.append(emb)
                    try:
                        import torch, gc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    except ImportError:
                        pass
                except Exception as batch_err:
                    logger.error(f"Error processing batch: {batch_err}")
                    continue
            
            if not embeddings_list:
                logger.error("No valid embeddings were generated for any batch")
                return []
            
            embeddings = np.concatenate(embeddings_list)
            
            expected_dim = getattr(self.embedding_func, 'embedding_dim', None)
            actual_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else None
            
            if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
                logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            
            for i, d in enumerate(list_data):
                if i < len(embeddings):
                    d["__vector__"] = embeddings[i]
                else:
                    logger.error(f"Index {i} out of bounds for embeddings with length {len(embeddings)}")
                    continue
            
            valid_list_data = [d for d in list_data if "__vector__" in d]
            if not valid_list_data:
                logger.error("No valid data with vectors to insert")
                return []
            
            client_dim = getattr(self._client, 'vec_dim', None)
            
            first_vector_dim = len(valid_list_data[0]["__vector__"])
            if client_dim is not None and client_dim != first_vector_dim:
                logger.warning(f"Vector dimension mismatch: client expects {client_dim}, got {first_vector_dim}")
                logger.info(f"Reinitializing vector database with new dimension: {first_vector_dim}")
                self._client = NanoVectorDB(
                    first_vector_dim, storage_file=self._client_file_name
                )
            
            try:
                results = self._client.upsert(datas=valid_list_data)
                return results
            except IndexError as e:
                logger.error(f"IndexError during upsert: {e}")
                logger.info("Attempting to fix and reinitialize vector database")
                
                self._client = NanoVectorDB(
                    first_vector_dim, storage_file=self._client_file_name
                )
                
                try:
                    results = self._client.upsert(datas=valid_list_data)
                    return results
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error during vector database upsert: {e}")
            return []

    async def query(self, query: str, top_k=5):
        try:
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
            
            client_dim = getattr(self._client, 'vec_dim', None)
            if client_dim is not None and len(embedding) != client_dim:
                logger.warning(f"Query vector dimension {len(embedding)} doesn't match database dimension {client_dim}")
                embedding = np.resize(embedding, (client_dim,))
            
            results = self._client.query(
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
            results = [
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
            ]
            return results
        except Exception as e:
            logger.error(f"Error during vector query: {e}")
            return []

    async def get_by_id(self, id_: str) -> Optional[dict]:
        try:
            if hasattr(self._client, 'get'):
                try:
                    result = self._client.get([id_])
                    if result and len(result) > 0:
                        data = result[0]
                        if data:
                            result_dict = {
                                "id": data.get("__id__", id_),
                                "distance": 0.0,
                            }
                            
                            for k, v in data.items():
                                if k not in ["__id__", "__vector__", "__metrics__"]:
                                    result_dict[k] = v
                            
                            return result_dict
                except Exception as e:
                    logger.debug(f"get method failed for ID {id_}: {e}")
            
            embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
            
            results = self._client.query(
                query=np.zeros(embed_dim),
                top_k=1000,
                better_than_threshold=0.0
            )
            
            for item in results:
                if item.get("__id__") == id_:
                    result = {
                        "id": item["__id__"],
                        "distance": item.get("__metrics__", 0.0),
                    }
                    
                    for k, v in item.items():
                        if k not in ["__id__", "__vector__", "__metrics__"]:
                            result[k] = v
                    
                    return result
            return None
        except Exception as e:
            logger.error(f"Error during get_by_id for ID {id_}: {e}")
            return None

    async def get_by_ids(self, ids: List[str]) -> List[dict]:
        try:
            if hasattr(self._client, 'get'):
                try:
                    records = self._client.get(ids)
                    results = []
                    
                    id_to_record = {record.get("__id__"): record for record in records if record}
                    
                    for id_ in ids:
                        if id_ in id_to_record:
                            record = id_to_record[id_]
                            result_dict = {
                                "id": record["__id__"],
                                "distance": 0.0,
                            }
                            
                            for k, v in record.items():
                                if k not in ["__id__", "__vector__", "__metrics__"]:
                                    result_dict[k] = v
                            
                            results.append(result_dict)
                        else:
                            results.append(None)
                    
                    return results
                except Exception as e:
                    logger.error(f"get method failed: {e}")
            
            results = []
            for id_ in ids:
                result = await self.get_by_id(id_)
                results.append(result)
                await asyncio.sleep(0)
            return results
        except Exception as e:
            logger.error(f"Error during get_by_ids: {e}")
            return [None] * len(ids)

    async def index_done_callback(self):
        try:
            self._client.save()
        except Exception as e:
            logger.error(f"Error during index_done_callback: {e}")
