from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rag_endpoint():
    response = client.post(
        "/rag",
        json={"question": "What does NVIDIA specialize in?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "relevant_chunks" in response.json()
    assert isinstance(response.json()["answer"], str)
    assert isinstance(response.json()["relevant_chunks"], list)

def test_rag_endpoint_invalid_input():
    response = client.post(
        "/rag",
        json={"invalid_key": "This is not a question"}
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_rag_endpoint_empty_question():
    response = client.post(
        "/rag",
        json={"question": ""}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "relevant_chunks" in response.json()
    # The exact behavior for an empty question might depend on your implementation