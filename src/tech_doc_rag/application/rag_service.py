from typing import List
from tech_doc_rag.domain.models import Citation, QueryRequest, QueryResponse
from tech_doc_rag.infrastructure.embeddings import LocalEmbeddingService
from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore
from tech_doc_rag.generation.groq_client import GroqLLMService


class RAGService:
    """End-to-End RAG Uygulama Servisi."""

    def __init__(self, vector_store: InMemoryVectorStore, embedder: LocalEmbeddingService):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_service = GroqLLMService()

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        # 1. Query Embedding
        query_emb = self.embedder.embed_text(request.query)

        # 2. Vector Search
        search_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )

        if not search_results:
            return QueryResponse(
                query=request.query,
                answer="Aradığınız konuyla ilgili dokümantasyonda yeterli benzerlikte içerik bulunamadı.",
                citations=[],
                retrieved_chunks_count=0
            )

        # 3. Context & Citations Hazırlığı
        context_blocks = []
        citations: List[Citation] = []

        for idx, res in enumerate(search_results, 1):
            chunk = res.chunk
            header = chunk.metadata.get("section_header", "Bilinmeyen Başlık")
            source_file = chunk.metadata.get("source_file", "Bilinmeyen Dosya")

            context_blocks.append(f"--- Kaynak {idx} [{header}] ---\n{chunk.content}")
            
            citations.append(
                Citation(
                    source_file=source_file,
                    section_header=header,
                    snippet=chunk.content[:150] + "..."
                )
            )

        full_context = "\n\n".join(context_blocks)

        # 4. LLM Generation
        llm_answer = self.llm_service.generate_response(
            query=request.query,
            context_text=full_context
        )

        return QueryResponse(
            query=request.query,
            answer=llm_answer,
            citations=citations,
            retrieved_chunks_count=len(search_results)
        )
