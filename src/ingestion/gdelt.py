import json
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from src.ingestion.adapters import generate_doc_id
from src.schemas import Document

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Macro / geopolitical / credit themes the engine monitors when no query is supplied.
DEFAULT_QUERIES = [
    '"federal reserve" OR "interest rates" OR inflation OR sanctions OR tariffs OR bankruptcy OR "credit rating" OR acquisition',
]


class GdeltAdapter:
    """
    Pulls live English-language headlines from the GDELT 2.0 DOC API (no API key required)
    and normalises them into `event_feed` Documents.

    GDELT throttles to roughly one request per five seconds, so calls are spaced and every
    successful response is persisted to a snapshot file. When the network is unavailable the
    adapter replays that snapshot, which keeps demos and tests deterministic.
    """

    def __init__(
        self,
        snapshot_path: str = "data/samples/gdelt_snapshot.json",
        min_interval_s: float = 5.5,
        timeout_s: float = 20.0,
    ):
        self.snapshot_path = snapshot_path
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_call = 0.0
        self.last_fetch_was_live = False

    def _throttle(self):
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def fetch_articles(
        self, query: str, max_records: int = 50, timespan: str = "1d", retries: int = 3
    ) -> List[dict]:
        params = {
            "query": f"({query}) sourcelang:english",
            "mode": "ArtList",
            "maxrecords": max_records,
            "timespan": timespan,
            "format": "json",
            "sort": "DateDesc",
        }
        headers = {"User-Agent": "risk-engine-hackathon/1.0"}
        for attempt in range(retries):
            self._throttle()
            resp = requests.get(GDELT_DOC_API, params=params, headers=headers, timeout=self.timeout_s)
            if resp.status_code == 429:
                time.sleep(self.min_interval_s * (attempt + 2))
                continue
            resp.raise_for_status()
            try:
                return resp.json().get("articles", [])
            except ValueError:
                # GDELT answers some malformed or throttled queries with plain text.
                return []
        return []

    @staticmethod
    def _parse_seendate(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

    def _to_documents(self, articles: List[dict], query: str) -> List[Document]:
        docs = []
        for art in articles:
            title = (art.get("title") or "").strip()
            if not title:
                continue
            docs.append(Document(
                doc_id=generate_doc_id("gdelt", art.get("url", title)),
                source="gdelt",
                source_type="event_feed",
                published_at=self._parse_seendate(art.get("seendate")),
                time_precision="minute",
                text=title,
                url=art.get("url"),
                raw_entity_hints=[],
                metadata={
                    "domain": art.get("domain"),
                    "source_country": art.get("sourcecountry"),
                    "query": query,
                },
            ))
        return docs

    def load_documents(
        self,
        queries: Optional[List[str]] = None,
        max_records: int = 50,
        timespan: str = "1d",
        allow_snapshot: bool = True,
    ) -> List[Document]:
        queries = queries or DEFAULT_QUERIES
        raw: List[dict] = []
        try:
            for q in queries:
                for art in self.fetch_articles(q, max_records=max_records, timespan=timespan):
                    art["_query"] = q
                    raw.append(art)
            self.last_fetch_was_live = bool(raw)
        except requests.RequestException:
            raw = []
            self.last_fetch_was_live = False

        if raw:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=1)
        elif allow_snapshot and os.path.exists(self.snapshot_path):
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        docs: List[Document] = []
        seen = set()
        for art in raw:
            for doc in self._to_documents([art], art.get("_query", "")):
                if doc.doc_id not in seen:
                    seen.add(doc.doc_id)
                    docs.append(doc)
        return docs
