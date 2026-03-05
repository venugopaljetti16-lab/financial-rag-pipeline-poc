"""
Document loader for financial data sources.
Handles ingestion of credit ratings reports, analyst notes, and structured financial data.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FinancialDocument:
    """Represents a financial document with metadata."""
    doc_id: str
    content: str
    doc_type: str  # "credit_rating", "analyst_note", "financial_statement"
    entity: str  # Company or issuer name
    date: str  # ISO format
    metadata: dict = field(default_factory=dict)


class FinancialDocumentLoader:
    """Loads and preprocesses financial documents for RAG pipeline."""

    VALID_DOC_TYPES = {"credit_rating", "analyst_note", "financial_statement", "research_report"}

    def __init__(self):
        self._documents: List[FinancialDocument] = []

    @property
    def documents(self) -> List[FinancialDocument]:
        return self._documents

    def load_from_json(self, path: str) -> List[FinancialDocument]:
        """Load documents from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return self.load_from_records(data)

    def load_from_records(self, records: List[dict]) -> List[FinancialDocument]:
        """Load documents from a list of dict records."""
        docs = []
        for rec in records:
            doc = self._parse_record(rec)
            if doc:
                docs.append(doc)
        self._documents.extend(docs)
        return docs

    def _parse_record(self, rec: dict) -> Optional[FinancialDocument]:
        """Parse a single record into a FinancialDocument."""
        required = {"doc_id", "content", "doc_type", "entity", "date"}
        if not required.issubset(rec.keys()):
            return None
        if rec["doc_type"] not in self.VALID_DOC_TYPES:
            return None
        if not rec["content"].strip():
            return None
        return FinancialDocument(
            doc_id=rec["doc_id"],
            content=rec["content"].strip(),
            doc_type=rec["doc_type"],
            entity=rec["entity"],
            date=rec["date"],
            metadata=rec.get("metadata", {}),
        )

    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
        """Split documents into overlapping chunks for embedding."""
        chunks = []
        for doc in self._documents:
            text = doc.content
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    "chunk_id": f"{doc.doc_id}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "entity": doc.entity,
                    "date": doc.date,
                    "metadata": doc.metadata,
                })
                start += chunk_size - overlap
                chunk_idx += 1
        return chunks

    def filter_by_type(self, doc_type: str) -> List[FinancialDocument]:
        """Filter loaded documents by type."""
        return [d for d in self._documents if d.doc_type == doc_type]

    def filter_by_entity(self, entity: str) -> List[FinancialDocument]:
        """Filter loaded documents by entity name (case-insensitive)."""
        return [d for d in self._documents if d.entity.lower() == entity.lower()]

    def get_stats(self) -> dict:
        """Return statistics about loaded documents."""
        type_counts = {}
        for doc in self._documents:
            type_counts[doc.doc_type] = type_counts.get(doc.doc_type, 0) + 1
        return {
            "total_documents": len(self._documents),
            "by_type": type_counts,
            "unique_entities": len(set(d.entity for d in self._documents)),
        }
