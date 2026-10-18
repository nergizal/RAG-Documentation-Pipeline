from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Sisteme yüklenen ham dokümanı temsil eder."""
    id: str = Field(..., description="Dokümanın benzersiz kimliği")
    content: str = Field(..., description="Dokümanın tam metin içeriği")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dokümana ait üst veriler")


class DocumentChunk(BaseModel):
    """Dokümanın parçalanmış (chunk edilmiş) halini temsil eder."""
    id: str = Field(..., description="Parçanın benzersiz kimliği")
    document_id: str = Field(..., description="Ait olduğu ana dokümanın kimliği")
    content: str = Field(..., description="Parçanın metin içeriği")
    chunk_index: int = Field(..., description="Doküman içindeki sırası")
    embedding: Optional[List[float]] = Field(default=None, description="Metnin vektör gömülüsü")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Parçaya özel üst veriler")


class SearchResult(BaseModel):
    """Retrieval aşamasında getirilen sonucu temsil eder."""
    chunk: DocumentChunk = Field(..., description="Eşleşen doküman parçası")
    score: float = Field(..., description="Benzerlik skoru")


class QueryRequest(BaseModel):
    """Kullanıcı sorgu isteğini temsil eder."""
    query: str = Field(..., min_length=3, description="Kullanıcının sorduğu soru")
    top_k: int = Field(default=3, ge=1, le=10, description="Getirilecek maksimum parça sayısı")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum benzerlik eşik skoru")


class Citation(BaseModel):
    """Üretilen cevabın dayandırıldığı kaynak bilgisini temsil eder."""
    source_file: str = Field(..., description="Kaynak dosya adı")
    section_header: Optional[str] = Field(default=None, description="İlgili başlık bilgisi")
    snippet: str = Field(..., description="Kullanılan metin kesiti")


class QueryResponse(BaseModel):
    """Kullanıcıya döndürülen nihai RAG yanıtını temsil eder."""
    query: str = Field(..., description="Orijinal kullanıcı sorusu")
    answer: str = Field(..., description="LLM tarafından üretilen yanıt")
    citations: List[Citation] = Field(default_factory=list, description="Yanıtta kullanılan kaynaklar")
    retrieved_chunks_count: int = Field(..., description="İşleme dahil edilen parça sayısı")