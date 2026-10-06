import io
import json
import zipfile

import requests

from src.ingestion.gdelt import GdeltAdapter

FAKE_ARTICLES = [
    {"url": "https://example.com/a", "title": "Fed signals further rate hikes as inflation persists",
     "seendate": "20261005T120000Z", "domain": "example.com", "sourcecountry": "United States"},
    {"url": "https://example.com/b", "title": "EU announces new sanctions package",
     "seendate": "20261005T130000Z", "domain": "example.com", "sourcecountry": "Belgium"},
    {"url": "https://example.com/c", "title": "", "seendate": "20261005T130000Z"},
]


def test_live_fetch_normalises_and_snapshots(tmp_path, monkeypatch):
    snapshot = tmp_path / "gdelt.json"
    adapter = GdeltAdapter(snapshot_path=str(snapshot), min_interval_s=0)
    monkeypatch.setattr(adapter, "fetch_articles", lambda q, **kw: [dict(a) for a in FAKE_ARTICLES])

    docs = adapter.load_documents(queries=["inflation"])

    assert adapter.last_fetch_was_live
    assert len(docs) == 2  # empty title dropped
    assert all(d.source_type == "event_feed" and d.source == "gdelt" for d in docs)
    assert docs[0].published_at.year == 2026
    assert snapshot.exists()


def test_falls_back_to_snapshot_when_offline(tmp_path, monkeypatch):
    snapshot = tmp_path / "gdelt.json"
    snapshot.write_text(json.dumps([dict(FAKE_ARTICLES[0], _query="inflation")]), encoding="utf-8")
    adapter = GdeltAdapter(snapshot_path=str(snapshot), min_interval_s=0)

    def offline(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(adapter, "fetch_articles", offline)
    monkeypatch.setattr(adapter, "fetch_gkg_articles", offline)
    docs = adapter.load_documents(queries=["inflation"])

    assert not adapter.last_fetch_was_live
    assert len(docs) == 1
    assert "rate hikes" in docs[0].text


def _gkg_row(title, themes, collection="1"):
    row = [""] * 27
    row[0], row[1], row[2], row[3], row[4] = "r1", "20261006074500", collection, "example.com", "https://example.com/x"
    row[7] = ";".join(themes)
    row[26] = f"<PAGE_LINKS></PAGE_LINKS><PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(row)


def test_gkg_parser_keeps_only_market_relevant_titles():
    rows = [
        _gkg_row("Central bank raises interest rates as inflation hits decade high", ["ECON_CENTRALBANK", "ECON_INFLATION"]),
        _gkg_row("Local bakery wins award for best sourdough in town", ["EPU_POLICY", "ECON_INFLATION"]),
        _gkg_row("Sanctions widen as armed conflict escalates near key oil routes", ["SANCTIONS", "ARMEDCONFLICT"],
                 collection="2"),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.gkg.csv", "\n".join(rows))

    arts = GdeltAdapter._parse_gkg(buf.getvalue())

    assert [a["title"] for a in arts] == ["Central bank raises interest rates as inflation hits decade high"]
    assert arts[0]["seendate"] == "20261006T074500Z"
