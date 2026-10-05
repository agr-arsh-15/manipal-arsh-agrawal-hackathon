import re
import yaml
from typing import List, Dict, Tuple, Optional


class EntityLinker:
    """
    Precision-driven Entity Linking and Disambiguation against a universe configuration.
    Features:
    - Exact ticker cashtag matching (e.g., $AAPL)
    - Full legal company name and canonical alias matching
    - Contextual negative disambiguation (blocks "Apple" if referring to fruit, "Amazon" for rainforest)
    - Short ticker safety (prevents bare 'KO', 'BA', 'DIS', 'ALL' from triggering without context)
    """

    FINANCE_CONTEXT_TERMS = {
        "stock", "shares", "share", "equity", "market", "nasdaq", "nyse", "investor",
        "investors", "q1", "q2", "q3", "q4", "earnings", "ceo", "cfo", "dividend",
        "revenue", "profit", "loss", "sec", "downgrade", "upgrade", "analyst", "analysts",
        "trading", "trade", "price", "valuation", "target"
    }

    def __init__(self, universe_config_path: str = "config/universe.yaml"):
        with open(universe_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.universe = data.get("universe", [])
        self._build_indexes()

    def _build_indexes(self):
        self.ticker_to_entity: Dict[str, dict] = {}
        self.alias_to_tickers: Dict[str, List[str]] = {}

        for entry in self.universe:
            ticker = entry["ticker"].upper()
            self.ticker_to_entity[ticker] = entry
            
            # Index aliases (case-insensitive)
            aliases = [entry["name"]] + entry.get("aliases", [])
            for alias in aliases:
                norm_alias = alias.strip().lower()
                if norm_alias not in self.alias_to_tickers:
                    self.alias_to_tickers[norm_alias] = []
                self.alias_to_tickers[norm_alias].append(ticker)

    def link(self, text: str, raw_hints: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Extracts linked tickers from text and hints.
        Returns a list of tuples: (ticker, company_name, confidence).
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b[a-z0-9_]+\b", text_lower))
        has_finance_context = bool(words.intersection(self.FINANCE_CONTEXT_TERMS))

        matched_tickers = set()

        # 1. Evaluate explicit external hints first
        if raw_hints:
            for hint in raw_hints:
                norm_hint = hint.strip().upper()
                if norm_hint in self.ticker_to_entity:
                    matched_tickers.add(norm_hint)

        # 2. Cashtag matching: $AAPL, $MSFT
        cashtags = re.findall(r"\$([A-Za-z]{1,5})\b", text)
        for tag in cashtags:
            tag_upper = tag.upper()
            if tag_upper in self.ticker_to_entity:
                matched_tickers.add(tag_upper)

        # 3. Canonical alias & name matching with negative filter checks
        for entry in self.universe:
            ticker = entry["ticker"]
            if ticker in matched_tickers:
                continue

            # Check negative contexts
            neg_contexts = entry.get("negative_contexts", [])
            is_blocked = False
            for neg in neg_contexts:
                if neg.lower() in text_lower:
                    is_blocked = True
                    break
            if is_blocked:
                continue

            # Check aliases
            aliases = [entry["name"]] + entry.get("aliases", [])
            for alias in aliases:
                norm_alias = alias.lower()
                # Exact word-boundary match for alias
                pattern = r"\b" + re.escape(norm_alias) + r"\b"
                if re.search(pattern, text_lower):
                    # Disambiguation safeguard for single common words
                    if len(norm_alias) <= 5 and not has_finance_context and not raw_hints:
                        continue
                    matched_tickers.add(ticker)
                    break

        # Format output
        results = []
        for ticker in sorted(matched_tickers):
            company_name = self.ticker_to_entity[ticker]["name"]
            confidence = 0.95 if (raw_hints or f"${ticker}" in text.upper()) else 0.85
            results.append((ticker, company_name, confidence))

        return results
