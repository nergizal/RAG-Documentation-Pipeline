from tech_doc_rag.retrieval.vector_store import InMemoryVectorStore


def test_vector_store_add_and_search(populated_vector_store):
    query_vector = [0.85, 0.88, 0.75]
    results = populated_vector_store.search(
        query_embedding=query_vector,
        top_k=1,
        similarity_threshold=0.5
    )

    assert len(results) == 1
    assert results[0].chunk.id == "chunk-2"
    assert results[0].score > 0.9


def test_vector_store_threshold_filtering(populated_vector_store):
    query_vector = [-1.0, -1.0, -1.0]
    results = populated_vector_store.search(
        query_embedding=query_vector,
        top_k=2,
        similarity_threshold=0.8
    )

    assert len(results) == 0
