import re
import yaml
from typing import Dict, List, Tuple, Optional


class EventTaxonomyLabeler:
    """
    Applies weak supervision rules to classify raw unstructured text into the 9 taxonomy classes.
    Follows priority rules and keyword heuristic matching.
    """

    def __init__(self, taxonomy_config_path: str = "config/taxonomy.yaml"):
        with open(taxonomy_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.classes = data.get("classes", [])
        self.class_names = [c["name"] for c in self.classes]
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled_rules = []
        for cls in self.classes:
            name = cls["name"]
            keywords = cls.get("keywords", [])
            if not keywords:
                continue
            
            # Construct word-boundary regex patterns
            escaped = [r"\b" + re.escape(kw.lower()) + r"\b" for kw in keywords]
            pattern = re.compile("|".join(escaped), re.IGNORECASE)
            self.compiled_rules.append((name, pattern))

    def label(self, text: str) -> Tuple[str, float]:
        """
        Labels text with weak event label and associated confidence.
        Returns: (event_type, event_confidence)
        """
        text_lower = text.lower()
        matched_classes = []

        for name, pattern in self.compiled_rules:
            matches = pattern.findall(text_lower)
            if matches:
                # Store class name and number of pattern hits
                matched_classes.append((name, len(matches)))

        if not matched_classes:
            return ("Other/None", 0.50)

        # Sort by frequency of matches
        matched_classes.sort(key=lambda x: x[1], reverse=True)
        top_class, hit_count = matched_classes[0]

        # Confidence heuristic based on agreement/hits
        confidence = min(0.70 + (0.10 * hit_count), 0.95)
        return (top_class, confidence)
