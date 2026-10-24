from pathlib import Path
from fastapi import FastAPI
from tech_doc_rag.api.v1.router import router as api_v1_router
from tech_doc_rag.api.dependencies import get_vector_store, get_embedding_service
from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers

app = FastAPI(
    title="Technical Documentation RAG API",
    description="Markdown dokümantasyonları için modüler ve kaynak gösterimli RAG servisi",
    version="1.0.0"
)

app.include_router(api_v1_router)


@app.on_event("startup")
def startup_event():
    """Uygulama başlarken varsayılan örnek dokümanı varsayılan olarak yükler."""
    sample_file = Path("data/raw/fastapi_tutorial.md")
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_markdown_by_headers(
            document_id="init-doc",
            content=content,
            source_file=sample_file.name
        )
        embedder = get_embedding_service()
        embedded_chunks = embedder.embed_chunks(chunks)

        store = get_vector_store()
        store.add_chunks(embedded_chunks)
        print(f"🚀 [Startup] {len(embedded_chunks)} varsayılan chunk vektör veritabanına yüklendi.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tech_doc_rag.main:app", host="0.0.0.0", port=8000, reload=True)