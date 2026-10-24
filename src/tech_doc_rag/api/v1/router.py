import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from tech_doc_rag.domain.models import QueryRequest, QueryResponse
from tech_doc_rag.api.dependencies import get_rag_service, get_vector_store, get_embedding_service
from tech_doc_rag.application.rag_service import RAGService
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore
from tech_doc_rag.infrastructure.embeddings import LocalEmbeddingService
from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])


class TextIngestRequest(BaseModel):
    content: str = Field(..., description="Markdown formatındaki ham metin")
    source_file: str = Field(default="api_input.md", description="Doküman adı")


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    message: str


@router.get("/health", tags=["System"])
def health_check():
    """Servis sağlık ve durum kontrolü."""
    return {"status": "healthy", "service": "Tech Doc RAG API"}


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text(
    payload: TextIngestRequest,
    vector_store: InMemoryVectorStore = Depends(get_vector_store),
    embedder: LocalEmbeddingService = Depends(get_embedding_service)
):
    """Metin formatındaki Markdown verisini işler, vektörleştirir ve depoya ekler."""
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Metin içeriği boş olamaz.")

    doc_id = str(uuid.uuid4())
    chunks = chunk_markdown_by_headers(
        document_id=doc_id,
        content=payload.content,
        source_file=payload.source_file
    )

    embedded_chunks = embedder.embed_chunks(chunks)
    vector_store.add_chunks(embedded_chunks)

    return IngestResponse(
        document_id=doc_id,
        chunks_created=len(embedded_chunks),
        message="Doküman başarıyla işlendi ve vektör veritabanına eklendi."
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    vector_store: InMemoryVectorStore = Depends(get_vector_store),
    embedder: LocalEmbeddingService = Depends(get_embedding_service)
):
    """Yüklenen .md dosyasını okur, işler ve depoya ekler."""
    if not file.filename.endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Sadece .md ve .txt dosyaları desteklenmektedir.")

    content = (await file.read()).decode("utf-8")
    doc_id = str(uuid.uuid4())

    chunks = chunk_markdown_by_headers(
        document_id=doc_id,
        content=content,
        source_file=file.filename
    )

    embedded_chunks = embedder.embed_chunks(chunks)
    vector_store.add_chunks(embedded_chunks)

    return IngestResponse(
        document_id=doc_id,
        chunks_created=len(embedded_chunks),
        message=f"'{file.filename}' dosyası başarıyla işlendi."
    )


@router.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Kullanıcı sorusuna vektör araması ve LLM üretimi ile yanıt döner."""
    return rag_service.answer_query(request)