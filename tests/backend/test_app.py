from backend.app import app

from fastapi.testclient import TestClient

# test that models are loaded correctly
def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "loaded_models": 10,
        }

# test that actions are returned correctly
def test_action():
    payload = {
        "observation": [0] * 100,
        "agent": "blue",
        "agent_id": 1,
        "opponent_id": 0,
    }

    with TestClient(app) as client:
        response = client.post("/api/action", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert type(data["action"]) is int
        assert 0 <= data["action"] <= 4


# test observation formatting failures
def test_observation_failure():
    payload = {
        "observation": [0]*6,
        "agent": "blue",
        "agent_id": 1,
        "opponent_id": 0,
    }
    
    with TestClient(app) as client:
        response = client.post("/api/action", json=payload)

        assert response.status_code == 422

def test_agent_failure():
    payload = {
        "observation": [0]*100,
        "agent": "bananas",
        "agent_id": 1,
        "opponent_id": 0,
    }
    
    with TestClient(app) as client:
        response = client.post("/api/action", json=payload)

        assert response.status_code == 422

def test_agent_id_failure():
    payload = {
        "observation": [0]*100,
        "agent": "blue",
        "agent_id": 12,
        "opponent_id": 0,
    }
    
    with TestClient(app) as client:
        response = client.post("/api/action", json=payload)

        assert response.status_code == 422

def test_opponent_id_failure():
    payload = {
        "observation": [0]*100,
        "agent": "blue",
        "agent_id": 1,
        "opponent_id": 56,
    }
    
    with TestClient(app) as client:
        response = client.post("/api/action", json=payload)

        assert response.status_code == 422


