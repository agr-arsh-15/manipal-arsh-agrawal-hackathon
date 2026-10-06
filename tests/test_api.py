import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.api.main as api

    monkeypatch.setattr(api, "SIGNALS_PATH", str(tmp_path / "signals.jsonl"))
    monkeypatch.setattr(api, "BACKEND", "baseline")
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["backend"] == "baseline" and body["signals_stored"] == 0


def test_analyze_persists_and_is_queryable(client):
    resp = client.post("/analyze", json={"items": [
        {"text": "Tesla stock slides as it recalls 100,000 vehicles over brake defect", "source_type": "news"},
        {"text": "Fed signals more rate hikes as inflation stays hot", "source_type": "event_feed"},
    ]})
    assert resp.status_code == 200
    signals = resp.json()
    assert len(signals) == 2
    for s in signals:
        assert {"sentiment_score", "event_type", "impact_score"} <= s.keys()

    assert client.get("/health").json()["signals_stored"] == 2
    tsla = client.get("/signals/TSLA").json()
    assert len(tsla) == 1 and tsla[0]["ticker"] == "TSLA"
    assert len(client.get("/signals", params={"limit": 1}).json()) == 1


def test_unknown_ticker_404(client):
    assert client.get("/signals/ZZZZ").status_code == 404


def test_validation_rejects_empty_request(client):
    assert client.post("/analyze", json={"items": []}).status_code == 422


def test_stress_endpoint_explicit_scenario(client):
    body = client.post("/modules/stress-test", json={"event_type": "Macroeconomic", "impact_score": 9}).json()
    assert body["triggered"] and body["total_pnl"] < 0
    assert body["cet1_ratio_after"] < body["cet1_ratio_before"]


def test_rebalancer_weights_sum_to_one(client):
    client.post("/analyze", json={"items": [{"text": "Apple shares soar on record earnings"}]})
    weights = client.get("/modules/rebalancer/weights").json()
    assert len(weights) == 20
    assert abs(sum(w["target_weight"] for w in weights) - 1) < 1e-3
