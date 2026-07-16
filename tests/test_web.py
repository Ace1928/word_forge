"""Unit tests for the Word Forge Web UI FastAPI application.

Exercises all endpoints including dashboard serving, graph retrieval,
fuzzy prefix word search, full word detail retrieval, and stats endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.database.database_manager import DBManager
from word_forge.web.app import WebApp


@pytest.fixture
def test_client(populated_db_manager: DBManager) -> Generator[TestClient, None, None]:
    """Fixture providing a TestClient configured with a populated DB."""
    # Initialize the emotion tables in the test DB
    from word_forge.emotion.emotion_manager import EmotionManager
    EmotionManager(populated_db_manager)

    # Instantiate the WebApp using the test database path
    webapp = WebApp(db_path=str(populated_db_manager.db_path))
    # Build graph explicitly to sync with test DB
    webapp.graph_manager.build_graph()

    with TestClient(webapp.app) as client:
        yield client


def test_dashboard_endpoint(test_client: TestClient) -> None:
    """Test that the index/dashboard serves successfully."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Word Forge" in response.text
    assert "static/css/app.css" in response.text


def test_graph_endpoint_focused(test_client: TestClient) -> None:
    """Test that /api/graph returns a focused neighborhood correctly."""
    response = test_client.get("/api/graph?focus=happiness&depth=1")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data

    # Verify nodes contain happiness and joy
    nodes = [node["wfTerm"] for node in data["nodes"]]
    assert "happiness" in nodes
    assert "joy" in nodes


def test_graph_endpoint_whole(test_client: TestClient) -> None:
    """Test that /api/graph returns the whole database graph when requested."""
    response = test_client.get("/api/graph?whole_graph=true")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


def test_graph_endpoint_not_found(test_client: TestClient) -> None:
    """Test that /api/graph returns 404 for missing focus word."""
    response = test_client.get("/api/graph?focus=nonexistentword")
    assert response.status_code == 404
    assert "nonexistentword" in response.json()["detail"]


def test_words_search_endpoint(test_client: TestClient) -> None:
    """Test that /api/words/search fuzzy-finds matches correctly."""
    response = test_client.get("/api/words/search?q=hap")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "hap"
    assert len(data["results"]) >= 1
    assert data["results"][0]["term"] == "happiness"


def test_words_detail_endpoint(test_client: TestClient) -> None:
    """Test that /api/words/{term} returns the full detail profile."""
    # Let's insert some mock emotion & pronunciation data for test.
    # Note: DBManager fixtures might be shared/isolated per fixture dependency.
    response = test_client.get("/api/words/happiness")
    assert response.status_code == 200
    data = response.json()
    assert data["term"] == "happiness"
    assert data["definition"] == "a state of joy"
    assert "pronunciations" in data
    assert "emotion" in data
    assert "relationships" in data
    assert "synonym" in data["relationships"]


def test_words_detail_endpoint_not_found(test_client: TestClient) -> None:
    """Test that /api/words/{term} returns 404 for nonexistent words."""
    response = test_client.get("/api/words/unknownword")
    assert response.status_code == 404


def test_stats_endpoint(test_client: TestClient) -> None:
    """Test that /api/stats returns database stats correctly."""
    response = test_client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_words"] == 8
    assert data["total_relationships"] == 5
    assert data["graph_nodes"] > 0
