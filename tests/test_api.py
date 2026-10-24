from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_chat_page_loads():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CloudRAG" in response.text


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(main, "collection_count", lambda: 0)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_requires_ingestion(monkeypatch):
    monkeypatch.setattr(main, "collection_count", lambda: 0)

    response = client.post(
        "/ask",
        json={"question": "What is the leave policy?"},
    )

    assert response.status_code == 400
    assert "ingest" in response.json()["detail"].lower()


def test_question_validation():
    response = client.post(
        "/ask",
        json={"question": "hi"},
    )

    assert response.status_code == 422


def test_whitespace_question_is_rejected():
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_top_k_must_be_in_range():
    too_small = client.post(
        "/ask",
        json={"question": "What is the leave policy?", "top_k": 0},
    )
    too_large = client.post(
        "/ask",
        json={"question": "What is the leave policy?", "top_k": 9},
    )

    assert too_small.status_code == 422
    assert too_large.status_code == 422


def test_security_question_preserves_prohibition(monkeypatch):
    monkeypatch.setattr(main, "collection_count", lambda: 1)

    monkeypatch.setattr(
        main,
        "search_chunks",
        lambda question, top_k: [
            {
                "source": "security_basics.txt",
                "chunk_index": 0,
                "distance": 0.1,
                "text": (
                    "Do not upload confidential documents to public AI "
                    "services without authorization."
                ),
            }
        ],
    )

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, context: (
            "No. Confidential documents must not be uploaded to public "
            "AI services without authorization."
        ),
    )

    response = client.post(
        "/ask",
        json={
            "question": (
                "Can confidential documents be uploaded to public AI "
                "services without authorization?"
            ),
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    answer = response.json()["answer"].lower()

    assert "without authorization" in answer
    assert "must not" in answer or "cannot" in answer or "do not" in answer
    assert not answer.startswith("yes")


def test_ollama_failure_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(main, "collection_count", lambda: 1)

    def fail_search(question, top_k):
        raise main.OllamaError("Ollama is unavailable.")

    monkeypatch.setattr(main, "search_chunks", fail_search)

    response = client.post(
        "/ask",
        json={"question": "What is the leave policy?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama is unavailable."
