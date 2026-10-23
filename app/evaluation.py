"""Lightweight, transparent evaluation helpers for the CloudRAG prototype."""
import re


def keyword_coverage(answer: str, required_keywords: list[str]) -> float:
    """Fraction of expected keyword phrases found in an answer (0.0 to 1.0)."""
    if not required_keywords:
        return 1.0
    normalized = re.sub(r"\s+", " ", answer.lower()).strip()
    matched = sum(1 for phrase in required_keywords if phrase.lower() in normalized)
    return matched / len(required_keywords)


def expected_source_retrieved(matches: list[dict], expected_source: str) -> float:
    """Return 1.0 if the expected source appears in retrieved chunks, else 0.0."""
    return float(any(item.get("source") == expected_source for item in matches))
