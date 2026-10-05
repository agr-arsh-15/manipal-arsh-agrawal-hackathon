import pytest
from datetime import datetime, timezone
from src.labeling.impact import MarketImpactEngine


@pytest.fixture
def impact_engine():
    return MarketImpactEngine(prices_dir="data/prices", config_path="config/engine.yaml")


def test_abnormal_move_calculation(impact_engine):
    # AAPL on a known historical trading day (2020-04-15)
    pub_date = datetime(2020, 4, 15, 14, 0, 0, tzinfo=timezone.utc)
    z = impact_engine.compute_abnormal_move("AAPL", pub_date)
    assert z is not None
    assert z >= 0.0
    
    score = impact_engine.map_to_score(z)
    assert 1 <= score <= 10


def test_weekend_rollover_leakage_safety(impact_engine):
    # Sunday publication (2020-04-19) must roll forward to Monday (2020-04-20)
    pub_date_weekend = datetime(2020, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
    z_weekend = impact_engine.compute_abnormal_move("AAPL", pub_date_weekend)
    
    pub_date_monday = datetime(2020, 4, 20, 9, 30, 0, tzinfo=timezone.utc)
    z_monday = impact_engine.compute_abnormal_move("AAPL", pub_date_monday)
    
    assert z_weekend is not None
    assert z_monday is not None
    # Both evaluate the same forward trading window
    assert abs(z_weekend - z_monday) < 1e-4
