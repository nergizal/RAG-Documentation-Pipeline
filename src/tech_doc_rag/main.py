from pathlib import Path
from tech_doc_rag.domain.models import QueryRequest
from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers
from tech_doc_rag.infrastructure.embeddings import LocalEmbeddingService
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore
from tech_doc_rag.application.rag_service import RAGService


def main():
    file_path = Path("data/raw/fastapi_tutorial.md")
    if not file_path.exists():
        print(f"Hata: {file_path} bulunamadı!")
        return

    # 1. Ingestion
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown_by_headers(
        document_id="doc-001",
        content=content,
        source_file=file_path.name
    )

    # 2. Embedding & Vector Store
    embedder = LocalEmbeddingService()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    vector_store = InMemoryVectorStore()
    vector_store.add_chunks(embedded_chunks)

    # 3. RAG Application Service Initializer
    rag_service = RAGService(vector_store=vector_store, embedder=embedder)

    # 4. Sorgu Testi
    query_text = "FastAPI kurulumu nasıl yapılır?"
    print(f"\n❓ Kullanıcı Sorusu: '{query_text}'\n")

    request = QueryRequest(query=query_text, top_k=2, similarity_threshold=0.2)
    response = rag_service.answer_query(request)

    print("=" * 60)
    print("🤖 RAG YANITI (LLM Generation + Citations)")
    print("=" * 60)
    print(f"\n{response.answer}\n")
    print("-" * 60)
    print(f"📚 Kullanılan Kaynak Sayısı: {response.retrieved_chunks_count}")
    for idx, cite in enumerate(response.citations, 1):
        print(f"  [{idx}] {cite.source_file} -> {cite.section_header}")


if __name__ == "__main__":
    main()
