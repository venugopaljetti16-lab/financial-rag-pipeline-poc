"""
Vector store for financial document embeddings.
Uses FAISS for efficient similarity search over document chunks.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class EmbeddingModel:
    """Lightweight embedding model interface.
    In production, this wraps sentence-transformers or OpenAI embeddings.
    For the POC, uses a deterministic hash-based embedding for testability.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding from text."""
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


class FinancialVectorStore:
    """FAISS-based vector store for financial document retrieval."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[dict] = []
        self._index = None

    @property
    def size(self) -> int:
        return len(self._metadata)

    def add_chunks(self, chunks: List[dict], embedding_model: EmbeddingModel) -> int:
        """Add document chunks to the vector store."""
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.embed_batch(texts)
        for emb, chunk in zip(embeddings, chunks):
            self._embeddings.append(emb)
            self._metadata.append(chunk)
        self._build_index()
        return len(chunks)

    def _build_index(self):
        """Build or rebuild the FAISS index."""
        try:
            import faiss
            matrix = np.array(self._embeddings, dtype=np.float32)
            self._index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(matrix)
            self._index.add(matrix)
        except ImportError:
            self._index = None

    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
        entity_filter: Optional[str] = None,
    ) -> List[Tuple[dict, float]]:
        """Search for similar chunks. Returns list of (metadata, score) tuples."""
        query_vec = embedding_model.embed(query).reshape(1, -1)

        if self._index is not None:
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self._index.search(query_vec, min(top_k * 3, self.size))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                meta = self._metadata[idx]
                if doc_type_filter and meta.get("doc_type") != doc_type_filter:
                    continue
                if entity_filter and meta.get("entity", "").lower() != entity_filter.lower():
                    continue
                results.append((meta, float(score)))
                if len(results) >= top_k:
                    break
            return results
        else:
            return self._brute_force_search(
                query_vec[0], top_k, doc_type_filter, entity_filter
            )

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        doc_type_filter: Optional[str],
        entity_filter: Optional[str],
    ) -> List[Tuple[dict, float]]:
        """Fallback brute-force cosine similarity search."""
        scores = []
        for i, emb in enumerate(self._embeddings):
            meta = self._metadata[i]
            if doc_type_filter and meta.get("doc_type") != doc_type_filter:
                continue
            if entity_filter and meta.get("entity", "").lower() != entity_filter.lower():
                continue
            cos_sim = float(np.dot(query_vec, emb) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-9
            ))
            scores.append((meta, cos_sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_stats(self) -> dict:
        """Return store statistics."""
        types = {}
        entities = set()
        for m in self._metadata:
            t = m.get("doc_type", "unknown")
            types[t] = types.get(t, 0) + 1
            entities.add(m.get("entity", ""))
        return {
            "total_chunks": self.size,
            "dimension": self.dimension,
            "has_faiss_index": self._index is not None,
            "by_type": types,
            "unique_entities": len(entities),
        }
