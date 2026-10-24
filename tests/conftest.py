import pytest
from tech_doc_rag.domain.models import DocumentChunk
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore


@pytest.fixture
def sample_markdown_content():
    return """# FastAPI Guide

FastAPI is a modern web framework for Python.

## Installation

Run pip install fastapi to install it.
"""


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="FastAPI is a modern web framework.",
            metadata={"section_header": "FastAPI Guide", "source_file": "test.md"},
            embedding=[0.1, 0.2, 0.3]
        ),
        DocumentChunk(
            id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            content="Run pip install fastapi to install.",
            metadata={"section_header": "Installation", "source_file": "test.md"},
            embedding=[0.8, 0.9, 0.7]
        )
    ]


@pytest.fixture
def populated_vector_store(sample_chunks):
    store = InMemoryVectorStore()
    store.add_chunks(sample_chunks)
    return store
