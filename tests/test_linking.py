import pytest
from src.linking.matcher import EntityLinker


@pytest.fixture
def linker():
    return EntityLinker("config/universe.yaml")


def test_cashtag_linking(linker):
    text = "Heavy trading volume seen on $AAPL and $MSFT ahead of earnings."
    matches = linker.link(text)
    tickers = [m[0] for m in matches]
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_alias_company_name_linking(linker):
    text = "Microsoft Corporation announced a strategic partnership with OpenAI."
    matches = linker.link(text)
    tickers = [m[0] for m in matches]
    assert "MSFT" in tickers


def test_negative_context_disambiguation(linker):
    # Apple fruit should be rejected
    text_fruit = "Doctors recommend eating an apple fruit every morning for dietary fiber."
    matches_fruit = linker.link(text_fruit)
    assert len(matches_fruit) == 0

    # Amazon rainforest should be rejected
    text_rainforest = "Deforestation rates in the amazon rainforest have dropped significantly."
    matches_rainforest = linker.link(text_rainforest)
    assert len(matches_rainforest) == 0


def test_positive_context_disambiguation(linker):
    text = "Apple released earnings showing strong iPhone revenue."
    matches = linker.link(text)
    tickers = [m[0] for m in matches]
    assert "AAPL" in tickers


def test_short_ticker_blocking_without_finance_context(linker):
    # 'dis' or 'ba' in casual English shouldn't trigger DIS / BA
    text_casual = "Why did he dis that person at the event?"
    matches = linker.link(text_casual)
    assert len(matches) == 0
