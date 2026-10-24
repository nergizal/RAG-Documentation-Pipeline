from functools import lru_cache
from tech_doc_rag.infrastructure.embeddings import LocalEmbeddingService
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore
from tech_doc_rag.application.rag_service import RAGService

# Uygulama boyunca tek bir VectorStore instance'ı tutulur
_global_vector_store = InMemoryVectorStore()


@lru_cache()
def get_embedding_service() -> LocalEmbeddingService:
    """Embedding modelini belleğe tek sefer yükler."""
    return LocalEmbeddingService()


def get_vector_store() -> InMemoryVectorStore:
    """Global bellek içi vektör veritabanını döndürür."""
    return _global_vector_store


def get_rag_service() -> RAGService:
    """RAG uygulama servisini hazırlar."""
    embedder = get_embedding_service()
    store = get_vector_store()
    return RAGService(vector_store=store, embedder=embedder)