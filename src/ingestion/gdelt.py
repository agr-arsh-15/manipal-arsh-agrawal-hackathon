import csv
import html
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from src.ingestion.adapters import generate_doc_id
from src.schemas import Document

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_RAW = "https://data.gdeltproject.org/gdeltv2"

# GKG themes that mark an article as market-relevant. Broad families such as EPU_* or ECON_* alone
# also tag lifestyle and local-politics stories, so a headline must hit at least two of these.
GKG_THEME_PREFIXES = (
    "ECON_STOCKMARKET", "ECON_INTEREST_RATE", "ECON_INFLATION", "ECON_BANKRUPTCY", "ECON_DEBT",
    "ECON_CENTRALBANK", "ECON_CURRENCY", "ECON_TRADE_DISPUTE", "ECON_OILPRICE", "ECON_EARNINGSREPORT",
    "ECON_IPO", "ECON_FREETRADE", "ECON_SUBPRIME", "ECON_UNEMPLOYMENT", "ECON_MONOPOLY",
    "EPU_CATS_MONETARY_POLICY", "EPU_CATS_TRADE_POLICY", "EPU_CATS_FINANCIAL_REGULATION",
    "SANCTIONS", "ARMEDCONFLICT", "CYBER_ATTACK", "WB_1104_MACROECONOMIC", "WB_318_FINANCIAL",
    "TAX_FNCACT_CENTRAL_BANK",
)
GKG_MIN_THEMES = 2
PAGE_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
csv.field_size_limit(sys.maxsize)

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
                if attempt + 1 < retries:
                    time.sleep(self.min_interval_s * (attempt + 2))
                continue
            resp.raise_for_status()
            try:
                return resp.json().get("articles", [])
            except ValueError:
                # GDELT answers some malformed or throttled queries with plain text.
                return []
        return []

    def fetch_gkg_articles(self, n_files: int = 8, max_records: int = 200) -> List[dict]:
        """
        Fallback that reads GDELT's raw 15-minute Global Knowledge Graph exports instead of the
        rate-limited DOC API. Keeps English articles tagged with market-relevant themes and uses
        their page titles as headlines. Returns records in the DOC API article format.
        """
        latest = requests.get(f"{GDELT_RAW}/lastupdate.txt", timeout=self.timeout_s)
        latest.raise_for_status()
        stamp = re.search(r"/(\d{14})\.gkg\.csv\.zip", latest.text)
        if not stamp:
            return []
        t = datetime.strptime(stamp.group(1), "%Y%m%d%H%M%S")
        articles: List[dict] = []
        for k in range(n_files):
            ts = (t - timedelta(minutes=15 * k)).strftime("%Y%m%d%H%M%S")
            resp = requests.get(f"{GDELT_RAW}/{ts}.gkg.csv.zip", timeout=self.timeout_s * 3)
            if resp.status_code != 200:
                continue
            articles += self._parse_gkg(resp.content)
            if len(articles) >= max_records:
                break
        seen, unique = set(), []
        for a in sorted(articles, key=lambda a: (a["_theme_hits"], a["seendate"]), reverse=True):
            if a["title"].lower() not in seen:
                seen.add(a["title"].lower())
                unique.append(a)
        return unique[:max_records]

    @staticmethod
    def _parse_gkg(zipped: bytes) -> List[dict]:
        out = []
        with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
        for row in csv.reader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 27 or row[2] != "1":  # collection 1 = web news
                continue
            themes = [th for th in row[7].split(";") if th]
            matched = sorted({th for th in themes if th.startswith(GKG_THEME_PREFIXES)})
            title_match = PAGE_TITLE.search(row[26])
            if len(matched) < GKG_MIN_THEMES or not title_match:
                continue
            title = " ".join(html.unescape(title_match.group(1)).split())
            if len(title.split()) < 6 or sum(ch.isascii() for ch in title) < 0.95 * len(title):
                continue
            out.append({
                "title": title,
                "url": row[4],
                "seendate": f"{row[1][:8]}T{row[1][8:]}Z",
                "domain": row[3],
                "sourcecountry": None,
                "_query": "gkg:" + ",".join(matched[:5]),
                "_theme_hits": len(matched),
            })
        return out

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
                for art in self.fetch_articles(q, max_records=max_records, timespan=timespan, retries=1):
                    art["_query"] = q
                    raw.append(art)
        except requests.RequestException:
            raw = []
        if not raw:
            try:
                raw = self.fetch_gkg_articles(max_records=max(max_records, 100))
            except (requests.RequestException, zipfile.BadZipFile):
                raw = []
        self.last_fetch_was_live = bool(raw)

        if raw:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=1)
        elif allow_snapshot and os.path.exists(self.snapshot_path):
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        return self._dedupe(raw)

    def load_snapshot(self) -> List[Document]:
        """Replays the last persisted GDELT pull without touching the network."""
        if not os.path.exists(self.snapshot_path):
            return []
        with open(self.snapshot_path, "r", encoding="utf-8") as f:
            return self._dedupe(json.load(f))

    def _dedupe(self, raw: List[dict]) -> List[Document]:
        docs: List[Document] = []
        seen = set()
        for art in raw:
            for doc in self._to_documents([art], art.get("_query", "")):
                if doc.doc_id not in seen:
                    seen.add(doc.doc_id)
                    docs.append(doc)
        return docs
