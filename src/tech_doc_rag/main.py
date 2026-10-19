from pathlib import Path
from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers
from tech_doc_rag.infrastructure.embeddings import LocalEmbeddingService
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore


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

    # 2. Embedding Generation
    print("Vektör gömülüleri (Embeddings) oluşturuluyor...")
    embedder = LocalEmbeddingService()
    embedded_chunks = embedder.embed_chunks(chunks)

    # 3. Vector Store'a Kayıt
    vector_store = InMemoryVectorStore()
    vector_store.add_chunks(embedded_chunks)
    print(f"{len(embedded_chunks)} adet chunk vektör veritabanına eklendi.\n")

    # 4. Arama Testi
    query = "How to install FastAPI?"
    print(f"🔎 Sorgu: '{query}'")
    
    query_embedding = embedder.embed_text(query)
    search_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=2,
        similarity_threshold=0.2
    )

    print("=" * 60)
    print(f" RETRIEVAL SONUÇLARI ({len(search_results)} eşleşme bulundu)")
    print("=" * 60)

    for idx, res in enumerate(search_results, 1):
        print(f"\n[{idx}] Skor: {res.score:.4f} | Header: {res.chunk.metadata.get('section_header')}")
        print(f"İçerik Kesiti: {res.chunk.content[:120]}...")


if __name__ == "__main__":
    main()