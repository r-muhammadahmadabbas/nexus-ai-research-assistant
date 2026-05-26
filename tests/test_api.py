"""
Unit tests for the FastAPI endpoints.
Tests health check, new conversation, and chat endpoint schemas.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data


class TestNewConversation:
    """Tests for the /new endpoint."""

    def test_new_returns_thread_id(self):
        response = client.post("/new")
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) > 0

    def test_new_returns_unique_ids(self):
        r1 = client.post("/new").json()
        r2 = client.post("/new").json()
        assert r1["thread_id"] != r2["thread_id"]


class TestChatEndpoint:
    """Tests for the /chat endpoint schema validation."""

    def test_chat_requires_message(self):
        """Chat endpoint should reject requests without a message field."""
        response = client.post("/chat", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_frontend_served(self):
        """Root endpoint should serve the frontend HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "").lower()
