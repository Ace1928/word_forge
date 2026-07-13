"""HTTP contract and security tests for the optional local GUI backend."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from word_forge.web.app import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path / "web.sqlite", api_token="test-token")
    with TestClient(app, headers={"X-Word-Forge-Token": "test-token"}) as test_client:
        yield test_client


def test_health_stats_and_entry_workflow(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in health.headers["content-security-policy"]

    created = client.post(
        "/api/entries",
        json={
            "term": "network",
            "definition": "Connected nodes and edges.",
            "language": "en",
        },
    )
    assert created.status_code == 201
    assert created.json()["term"] == "network"

    entry = client.get("/api/entry", params={"term": "network", "language": "en"})
    assert entry.status_code == 200
    assert entry.json()["definition"] == "Connected nodes and edges."

    search = client.get("/api/entries", params={"query": "nodes"})
    assert [item["term"] for item in search.json()] == ["network"]
    assert client.get("/api/stats").json()["words"] == 1


def test_network_and_mutation_events(client: TestClient) -> None:
    with client.websocket_connect(
        "/api/events", subprotocols=["word-forge", "test-token"]
    ) as websocket:
        assert websocket.receive_json()["type"] == "ready"
        response = client.post("/api/entries", json={"term": "graph"})
        assert response.status_code == 201
        assert websocket.receive_json() == {
            "type": "entry.changed",
            "term": "graph",
            "language": "en",
            "sequence": 1,
        }

    network = client.get("/api/network")
    assert network.status_code == 200
    assert network.json()["nodes"][0]["term"] == "graph"


def test_validation_missing_entry_and_origin_protection(client: TestClient) -> None:
    assert client.post("/api/entries", json={"term": ""}).status_code == 422
    assert (
        client.post(
            "/api/entries",
            json={"term": "strict", "language": "en", "defintion": "misspelled"},
        ).status_code
        == 422
    )

    missing = client.get("/api/entry", params={"term": "missing"})
    assert missing.status_code == 404
    assert missing.json()["error"] == "entry_not_found"

    rejected = client.post(
        "/api/entries",
        headers={"Origin": "https://attacker.example"},
        json={"term": "unsafe"},
    )
    assert rejected.status_code == 403
    assert rejected.json()["error"] == "cross_origin_request"


def test_static_workspace_and_openapi_are_available(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Word Forge" in page.text
    assert client.get("/api/openapi.json").status_code == 200


def test_revision_conflicts_and_sense_lifecycle(client: TestClient) -> None:
    created = client.post(
        "/api/entries",
        json={"term": "bank", "expected_revision": 0},
    )
    assert created.status_code == 201
    revision = created.json()["revision"]
    updated = client.post(
        "/api/entries",
        json={"term": "bank", "definition": "Current", "expected_revision": revision},
    )
    assert updated.status_code == 201
    stale = client.post(
        "/api/entries",
        json={"term": "bank", "definition": "Stale", "expected_revision": revision},
    )
    assert stale.status_code == 409
    assert stale.json()["error"] == "revision_conflict"

    sense = client.post(
        "/api/senses",
        json={
            "term": "bank",
            "definition": "Land beside a river.",
            "domain": "geography",
            "expected_revision": 0,
        },
    )
    assert sense.status_code == 201
    sense_payload = sense.json()
    entry = client.get("/api/entry", params={"term": "bank"}).json()
    assert entry["senses"][0]["id"] == sense_payload["id"]
    deleted = client.delete(
        f"/api/senses/{sense_payload['id']}",
        params={"expected_revision": sense_payload["revision"]},
    )
    assert deleted.json() == {"deleted": True}
    assert client.get("/api/entry", params={"term": "bank"}).json()["senses"] == []


def test_gui_integrity_backup_and_confirmed_restore(client: TestClient) -> None:
    created = client.post(
        "/api/entries",
        json={"term": "snapshot", "definition": "Before", "expected_revision": 0},
    ).json()
    integrity = client.get("/api/maintenance/integrity")
    assert integrity.status_code == 200
    assert integrity.json()["ok"] is True

    backup = client.post("/api/maintenance/backups", json={"label": "before-edit"})
    assert backup.status_code == 201
    backup_name = backup.json()["name"]
    assert backup_name.endswith("-before-edit.sqlite")
    assert [item["name"] for item in client.get("/api/maintenance/backups").json()] == [backup_name]

    updated = client.post(
        "/api/entries",
        json={
            "term": "snapshot",
            "definition": "After",
            "expected_revision": created["revision"],
        },
    )
    assert updated.json()["definition"] == "After"
    rejected = client.post(
        "/api/maintenance/restore",
        json={"backup_name": backup_name, "confirmation": "no"},
    )
    assert rejected.status_code == 400

    restored = client.post(
        "/api/maintenance/restore",
        json={"backup_name": backup_name, "confirmation": "RESTORE"},
    )
    assert restored.json() == {"restored": True, "backup_name": backup_name}
    entry = client.get("/api/entry", params={"term": "snapshot"})
    assert entry.json()["definition"] == "Before"


def test_managed_backup_names_are_unique_and_paths_are_bounded(client: TestClient) -> None:
    client.post("/api/entries", json={"term": "bounded"})
    first = client.post("/api/maintenance/backups", json={"label": "release"})
    second = client.post("/api/maintenance/backups", json={"label": "release"})
    assert first.status_code == second.status_code == 201
    assert first.json()["name"] != second.json()["name"]

    traversal = client.post(
        "/api/maintenance/restore",
        json={"backup_name": "../outside.sqlite", "confirmation": "RESTORE"},
    )
    assert traversal.status_code == 400


def test_api_and_websocket_require_the_session_secret(tmp_path: Path) -> None:
    app = create_app(tmp_path / "secured.sqlite", api_token="expected-token")
    with TestClient(app) as unauthorized:
        response = unauthorized.get("/api/health")
        assert response.status_code == 401
        assert response.json()["error"] == "invalid_session"

        with pytest.raises(WebSocketDisconnect):
            with unauthorized.websocket_connect(
                "/api/events", subprotocols=["word-forge", "wrong-token"]
            ):
                pass


def test_host_and_websocket_origin_are_restricted(client: TestClient) -> None:
    rejected_host = client.get("/api/health", headers={"Host": "attacker.example"})
    assert rejected_host.status_code == 400

    with pytest.raises(WebSocketDisconnect) as rejected_origin:
        with client.websocket_connect(
            "/api/events",
            headers={"Origin": "https://attacker.example"},
            subprotocols=["word-forge", "test-token"],
        ):
            pass
    assert rejected_origin.value.code == 1008
