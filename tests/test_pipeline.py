"""Tests for the Financial RAG Pipeline."""

import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import FinancialDocumentLoader, FinancialDocument
from src.vector_store import EmbeddingModel, FinancialVectorStore
from src.rag_pipeline import FinancialRAGPipeline


# ── Sample Data ──────────────────────────────────────────────

SAMPLE_RECORDS = [
    {
        "doc_id": "CR-001",
        "content": "S&P has affirmed AA- rating on Acme Energy Corp with stable outlook. Debt-to-EBITDA 2.3x.",
        "doc_type": "credit_rating",
        "entity": "Acme Energy Corp",
        "date": "2026-01-15",
        "metadata": {"rating": "AA-"},
    },
    {
        "doc_id": "AN-001",
        "content": "Acme Energy Q4 earnings exceeded estimates by 12%. Revenue grew 8.4% YoY to $18.7B.",
        "doc_type": "analyst_note",
        "entity": "Acme Energy Corp",
        "date": "2026-02-10",
        "metadata": {"recommendation": "Buy"},
    },
    {
        "doc_id": "CR-002",
        "content": "Moody's downgraded Global Trade Finance from Baa1 to Baa2 with negative outlook.",
        "doc_type": "credit_rating",
        "entity": "Global Trade Finance Ltd",
        "date": "2026-02-01",
        "metadata": {"rating": "Baa2"},
    },
]


# ── Document Loader Tests ────────────────────────────────────


class TestDocumentLoader:
    def test_load_records(self):
        loader = FinancialDocumentLoader()
        docs = loader.load_from_records(SAMPLE_RECORDS)
        assert len(docs) == 3

    def test_document_fields(self):
        loader = FinancialDocumentLoader()
        docs = loader.load_from_records(SAMPLE_RECORDS)
        assert docs[0].doc_id == "CR-001"
        assert docs[0].entity == "Acme Energy Corp"
        assert docs[0].doc_type == "credit_rating"

    def test_reject_invalid_doc_type(self):
        loader = FinancialDocumentLoader()
        bad = [{"doc_id": "X", "content": "test", "doc_type": "invalid", "entity": "A", "date": "2026-01-01"}]
        docs = loader.load_from_records(bad)
        assert len(docs) == 0

    def test_reject_empty_content(self):
        loader = FinancialDocumentLoader()
        bad = [{"doc_id": "X", "content": "   ", "doc_type": "credit_rating", "entity": "A", "date": "2026-01-01"}]
        docs = loader.load_from_records(bad)
        assert len(docs) == 0

    def test_reject_missing_fields(self):
        loader = FinancialDocumentLoader()
        bad = [{"doc_id": "X", "content": "test"}]
        docs = loader.load_from_records(bad)
        assert len(docs) == 0

    def test_chunk_documents(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        chunks = loader.chunk_documents(chunk_size=50, overlap=10)
        assert len(chunks) > len(SAMPLE_RECORDS)
        assert all("chunk_id" in c for c in chunks)
        assert all("text" in c for c in chunks)

    def test_chunk_preserves_metadata(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        chunks = loader.chunk_documents(chunk_size=50, overlap=10)
        first = chunks[0]
        assert "entity" in first
        assert "doc_type" in first
        assert "doc_id" in first

    def test_filter_by_type(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        ratings = loader.filter_by_type("credit_rating")
        assert len(ratings) == 2
        assert all(d.doc_type == "credit_rating" for d in ratings)

    def test_filter_by_entity(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        acme = loader.filter_by_entity("Acme Energy Corp")
        assert len(acme) == 2

    def test_get_stats(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        stats = loader.get_stats()
        assert stats["total_documents"] == 3
        assert stats["unique_entities"] == 2
        assert "credit_rating" in stats["by_type"]


# ── Embedding Model Tests ────────────────────────────────────


class TestEmbeddingModel:
    def test_embed_returns_correct_dimension(self):
        model = EmbeddingModel(dimension=384)
        vec = model.embed("test text")
        assert vec.shape == (384,)

    def test_embed_is_deterministic(self):
        model = EmbeddingModel()
        v1 = model.embed("same text")
        v2 = model.embed("same text")
        assert (v1 == v2).all()

    def test_different_text_different_embedding(self):
        model = EmbeddingModel()
        v1 = model.embed("text one")
        v2 = model.embed("text two")
        assert not (v1 == v2).all()

    def test_embed_batch(self):
        model = EmbeddingModel(dimension=128)
        vecs = model.embed_batch(["a", "b", "c"])
        assert vecs.shape == (3, 128)

    def test_embedding_is_normalized(self):
        import numpy as np
        model = EmbeddingModel()
        vec = model.embed("normalize me")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01


# ── Vector Store Tests ───────────────────────────────────────


class TestVectorStore:
    def _make_chunks(self):
        loader = FinancialDocumentLoader()
        loader.load_from_records(SAMPLE_RECORDS)
        return loader.chunk_documents(chunk_size=100, overlap=20)

    def test_add_chunks(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        added = store.add_chunks(chunks, model)
        assert added == len(chunks)
        assert store.size == len(chunks)

    def test_search_returns_results(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        store.add_chunks(chunks, model)
        results = store.search("credit rating energy", model, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_returns_scores(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        store.add_chunks(chunks, model)
        results = store.search("Acme Energy", model, top_k=2)
        for meta, score in results:
            assert isinstance(score, float)
            assert "chunk_id" in meta

    def test_search_entity_filter(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        store.add_chunks(chunks, model)
        results = store.search("rating", model, top_k=10, entity_filter="Acme Energy Corp")
        assert all(m["entity"] == "Acme Energy Corp" for m, _ in results)

    def test_search_doc_type_filter(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        store.add_chunks(chunks, model)
        results = store.search("Q4 earnings", model, top_k=10, doc_type_filter="analyst_note")
        assert all(m["doc_type"] == "analyst_note" for m, _ in results)

    def test_get_stats(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        chunks = self._make_chunks()
        store.add_chunks(chunks, model)
        stats = store.get_stats()
        assert stats["total_chunks"] == len(chunks)
        assert stats["dimension"] == 384

    def test_empty_search(self):
        store = FinancialVectorStore()
        model = EmbeddingModel()
        results = store.search("anything", model)
        assert results == []


# ── RAG Pipeline Tests ───────────────────────────────────────


class TestRAGPipeline:
    def test_ingest(self):
        pipe = FinancialRAGPipeline()
        result = pipe.ingest(SAMPLE_RECORDS)
        assert result["documents_loaded"] == 3
        assert result["chunks_created"] > 0
        assert pipe.is_initialized

    def test_query_before_ingest(self):
        pipe = FinancialRAGPipeline()
        resp = pipe.query("What is the rating?")
        assert resp.confidence == 0.0
        assert "not initialized" in resp.answer.lower()

    def test_query_returns_response(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        resp = pipe.query("What is the credit rating for Acme Energy?")
        assert resp.query == "What is the credit rating for Acme Energy?"
        assert len(resp.answer) > 0
        assert len(resp.sources) > 0

    def test_query_with_entity_filter(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        resp = pipe.query("credit outlook", entity_filter="Global Trade Finance Ltd")
        for src in resp.sources:
            assert src["entity"] == "Global Trade Finance Ltd"

    def test_query_with_doc_type_filter(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        resp = pipe.query("earnings revenue", doc_type_filter="analyst_note")
        for src in resp.sources:
            assert src["doc_type"] == "analyst_note"

    def test_response_has_sources(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        resp = pipe.query("Acme Energy rating")
        assert all("chunk_id" in s for s in resp.sources)
        assert all("score" in s for s in resp.sources)

    def test_pipeline_stats(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        stats = pipe.get_pipeline_stats()
        assert stats["is_initialized"]
        assert stats["document_stats"]["total_documents"] == 3
        assert stats["vector_store_stats"]["total_chunks"] > 0

    def test_confidence_in_range(self):
        pipe = FinancialRAGPipeline()
        pipe.ingest(SAMPLE_RECORDS)
        resp = pipe.query("Acme Energy Corp credit rating")
        assert 0.0 <= resp.confidence <= 1.0


# ── Integration Test ─────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline_flow(self):
        """End-to-end: load data -> ingest -> query -> validate response."""
        data_path = Path(__file__).parent.parent / "data" / "sample_financial_docs.json"
        with open(data_path) as f:
            records = json.load(f)

        pipe = FinancialRAGPipeline(chunk_size=300, chunk_overlap=50, top_k=3)
        result = pipe.ingest(records)
        assert result["documents_loaded"] == 7
        assert pipe.is_initialized

        resp = pipe.query("What is the credit rating and outlook for Acme Energy Corp?")
        assert len(resp.sources) > 0
        assert resp.context_used > 0
        assert len(resp.answer) > 50

        stats = pipe.get_pipeline_stats()
        assert stats["document_stats"]["unique_entities"] >= 3
