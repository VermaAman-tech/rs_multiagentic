import pytest
from fastapi.testclient import TestClient
from tools.server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["n_tools"] == 28

def test_object_detection():
    resp = client.post("/tools/ObjectDetection", json={"image_path": "a.jpg", "text_prompt": "car"})
    assert resp.status_code == 200
    assert "bboxes" in resp.json()

def test_get_area_boundary():
    resp = client.post("/tools/GetAreaBoundary", json={"area_name": "Texas"})
    assert resp.status_code == 200
    assert "boundary_wkt" in resp.json()

def test_calculator():
    resp = client.post("/tools/Calculator", json={"expression": "2+2"})
    assert resp.status_code == 200
    assert resp.json()["result"] == 4.0

def test_search():
    resp = client.post("/tools/GoogleSearch", json={"query": "flood"})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) > 0

