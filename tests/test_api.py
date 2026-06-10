import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_appointment_question():
    response = client.post("/ask", json={"question": "Can I book a cardiology appointment for Monday?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "tool_used" in data
    assert data["tool_used"] == "appointment_scheduler"
    assert "cardiology" in data["answer"].lower() or "Cardiology" in data["answer"]


def test_ask_returns_expected_keys():
    response = client.post("/ask", json={"question": "What are the signs of infection after discharge?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
