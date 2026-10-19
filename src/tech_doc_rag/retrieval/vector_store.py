from typing import List
import numpy as np
from tech_doc_rag.domain.models import DocumentChunk, SearchResult


class InMemoryVectorStore:
    """Bellek içi vektör veritabanı ve Kosinüs Benzerliği arama motoru."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Vektörleri üretilmiş chunk'ları depoya ekler."""
        for chunk in chunks:
            if chunk.embedding is not None:
                self.chunks.append(chunk)

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 3, 
        similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Sorgu vektörü ile depodaki tüm chunk vektörleri arasındaki Kosinüs Benzerliğini hesaplar,
        skor eşiğini geçen en yüksek top_k sonucu döndürür.
        """
        if not self.chunks:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        doc_vecs = np.array([c.embedding for c in self.chunks], dtype=np.float32)

        # Vectorized Cosine Similarity
        dot_product = np.dot(doc_vecs, q_vec)
        q_norm = np.linalg.norm(q_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        # Sıfıra bölünme hatasını engelleme
        scores = dot_product / (doc_norms * q_norm + 1e-10)

        results: List[SearchResult] = []
        for chunk, score in zip(self.chunks, scores):
            score_float = float(score)
            if score_float >= similarity_threshold:
                results.append(SearchResult(chunk=chunk, score=score_float))

        # Skorlara göre büyükten küçüğe sırala
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]