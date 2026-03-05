# Financial RAG Pipeline POC

A production-ready Retrieval-Augmented Generation (RAG) pipeline for querying proprietary financial datasets including credit ratings, analyst notes, and financial statements.

## Architecture

```
Query -> Embedding Model -> FAISS Vector Search -> Context Assembly -> LLM Response
```

**Components:**
- **Document Loader** — Ingests financial documents (credit ratings, analyst notes, financial statements, research reports) with metadata preservation and configurable chunking
- **Vector Store** — FAISS-backed similarity search with entity and document type filtering
- **RAG Pipeline** — End-to-end orchestration with structured response generation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | sentence-transformers / FAISS |
| Orchestration | LangChain-compatible architecture |
| Vector DB | FAISS (CPU), extensible to Pinecone/Weaviate |
| Language | Python 3.10+ |
| Testing | pytest (30 tests, 100% pass) |

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Usage

```python
from src.rag_pipeline import FinancialRAGPipeline
import json

# Load financial documents
with open("data/sample_financial_docs.json") as f:
    records = json.load(f)

# Initialize and ingest
pipeline = FinancialRAGPipeline(chunk_size=300, top_k=3)
pipeline.ingest(records)

# Query with optional filters
response = pipeline.query(
    "What is the credit rating for Acme Energy Corp?",
    entity_filter="Acme Energy Corp"
)

print(response.answer)
print(f"Sources: {len(response.sources)}, Confidence: {response.confidence}")
```

## Features

- **Financial document types**: Credit ratings, analyst notes, financial statements, research reports
- **Metadata-aware retrieval**: Filter by entity, document type, date
- **Configurable chunking**: Adjustable chunk size and overlap for optimal retrieval
- **Production patterns**: Structured responses, confidence scoring, source attribution
- **Extensible**: Swap embedding model, vector DB, or LLM without changing pipeline logic

## Project Structure

```
financial-rag-pipeline-poc/
├── src/
│   ├── document_loader.py    # Financial document ingestion & chunking
│   ├── vector_store.py       # FAISS vector search with filtering
│   └── rag_pipeline.py       # End-to-end RAG orchestration
├── tests/
│   └── test_pipeline.py      # 30 tests covering all components
├── data/
│   └── sample_financial_docs.json  # 7 sample financial documents
├── requirements.txt
└── README.md
```

## Author

**Venu Gopal Jetti** — Senior AI/ML Engineer
- GitHub: [venugopaljetti16-lab](https://github.com/venugopaljetti16-lab)
- LinkedIn: [venu-gopal-jetti](https://www.linkedin.com/in/venu-gopal-jetti-6a412124b/)
