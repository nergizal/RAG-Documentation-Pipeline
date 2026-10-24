from fastapi.testclient import TestClient
from tech_doc_rag.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ingest_text_endpoint():
    payload = {
        "content": "# Test Header\nThis is a test chunk content for API testing.",
        "source_file": "api_test.md"
    }
    response = client.post("/api/v1/ingest/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] > 0
    assert "document_id" in data


def test_query_endpoint():
    payload = {
        "query": "test chunk content",
        "top_k": 1,
        "similarity_threshold": 0.1
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
