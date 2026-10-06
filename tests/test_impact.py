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


def test_market_move_ranks_crash_days_above_calm_days(impact_engine):
    # 2020-03-16 (S&P 500 -12%) must dwarf a quiet session such as 2019-07-03
    crash = impact_engine.compute_market_move(datetime(2020, 3, 16, 14, 0, tzinfo=timezone.utc))
    calm = impact_engine.compute_market_move(datetime(2019, 7, 3, 14, 0, tzinfo=timezone.utc))
    assert crash is not None and calm is not None
    assert crash > 1.0 > calm


def test_post_close_publication_rolls_to_next_session(impact_engine):
    # 17:30 New York on Wed 2020-04-15 cannot be priced until Thu 2020-04-16
    after_close = datetime(2020, 4, 15, 21, 30, tzinfo=timezone.utc)
    next_open = datetime(2020, 4, 16, 14, 0, tzinfo=timezone.utc)
    assert MarketImpactEngine.effective_event_date(after_close).isoformat() == "2020-04-16"
    assert abs(
        impact_engine.compute_abnormal_move("AAPL", after_close)
        - impact_engine.compute_abnormal_move("AAPL", next_open)
    ) < 1e-9
