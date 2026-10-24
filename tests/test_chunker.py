from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers


def test_chunk_markdown_by_headers_creates_correct_chunks(sample_markdown_content):
    chunks = chunk_markdown_by_headers(
        document_id="doc-test",
        content=sample_markdown_content,
        source_file="test.md"
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["section_header"] == "FastAPI Guide"
    assert chunks[1].metadata["section_header"] == "FastAPI Guide > Installation"
    assert "pip install" in chunks[1].content


def test_chunk_markdown_empty_content():
    chunks = chunk_markdown_by_headers(
        document_id="doc-empty",
        content="",
        source_file="empty.md"
    )
    assert len(chunks) == 0
