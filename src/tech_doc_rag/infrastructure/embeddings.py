from typing import List
from sentence_transformers import SentenceTransformer
from tech_doc_rag.domain.models import DocumentChunk


class LocalEmbeddingService:
    """SentenceTransformers kütüphanesini kullanarak metinleri vektörlere dönüştürür."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Model ilk çağrıldığında hafızaya yüklenir
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Tekil bir metin için 384 boyutlu float vektör döndürür."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Verilen chunk listesindeki metinlerin embedding alanlarını doldurur."""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        return chunks