from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def _local_route(spam_probability: float = 0.1) -> dict:
    return {
        "route": "url_spam",
        "classification": {
            "model": "distilbert_m2",
            "available": True,
            "label": "LABEL_0",
            "confidence": 0.9,
            "route_hint": "url_spam",
        },
        "result": {
            "model": "lighturlnet",
            "available": True,
            "spam_probability": spam_probability,
            "spam": spam_probability >= 0.5,
            "detected_url": "https://example.com",
        },
    }


def _rdap_ok() -> dict:
    return {
        "source": "rdap",
        "checked": True,
        "found": True,
        "domain": "example.com",
        "lookup_url": "https://rdap.org/domain/example.com",
        "domain_age_days": 9000,
        "statuses": [],
    }


def _knowledge_ok() -> dict:
    return {
        "source": "local_intel_corpus",
        "checked": True,
        "embedding_available": True,
        "matches": [
            {
                "chunk_id": "intel-safe-browsing-001",
                "title": "Safe Browsing interpretation",
                "signals": ["no Safe Browsing match is not proof that a URL is safe"],
            }
        ],
    }


class DecisionToolTests(unittest.TestCase):
    def test_safe_browsing_match_forces_high_risk(self) -> None:
        with (
            patch.object(main, "route_request", return_value=_local_route(0.1)),
            patch.object(main, "_lookup_rdap", return_value=_rdap_ok()),
            patch.object(
                main,
                "_check_safe_browsing",
                return_value={
                    "source": "google_safe_browsing",
                    "checked": True,
                    "matched": True,
                    "matches": [{"threatType": "SOCIAL_ENGINEERING"}],
                },
            ),
            patch.object(main, "_retrieve_intel_articles", return_value=_knowledge_ok()),
        ):
            result = main.decide_url_threat("https://example.com")

        self.assertEqual(result["verdict"], "high_risk")
        self.assertGreaterEqual(result["risk_score"], 0.9)
        self.assertEqual(result["unknowns"], [])

    def test_rdap_timeout_keeps_result_in_review(self) -> None:
        with (
            patch.object(main, "route_request", return_value=_local_route(0.1)),
            patch.object(
                main,
                "_lookup_rdap",
                return_value={
                    "source": "rdap",
                    "checked": False,
                    "domain": "example.com",
                    "error": "timed out",
                },
            ),
            patch.object(
                main,
                "_check_safe_browsing",
                return_value={
                    "source": "google_safe_browsing",
                    "checked": True,
                    "matched": False,
                    "matches": [],
                },
            ),
            patch.object(main, "_retrieve_intel_articles", return_value=_knowledge_ok()),
        ):
            result = main.decide_url_threat("https://example.com")

        self.assertEqual(result["verdict"], "needs_review")
        self.assertTrue(any("RDAP was not checked" in item for item in result["unknowns"]))

    def test_suspicious_local_model_needs_or_high_risk_with_citations(self) -> None:
        with (
            patch.object(main, "route_request", return_value=_local_route(0.62)),
            patch.object(main, "_lookup_rdap", return_value=_rdap_ok()),
            patch.object(
                main,
                "_check_safe_browsing",
                return_value={
                    "source": "google_safe_browsing",
                    "checked": True,
                    "matched": False,
                    "matches": [],
                },
            ),
            patch.object(main, "_retrieve_intel_articles", return_value=_knowledge_ok()),
        ):
            result = main.decide_url_threat("urgent payment required https://example.com/login")

        self.assertIn(result["verdict"], {"needs_review", "high_risk"})
        self.assertTrue(result["citations"])
        self.assertTrue(result["evidence"]["knowledge"]["matches"])

    def test_missing_safe_browsing_without_other_risk_needs_review(self) -> None:
        with (
            patch.object(main, "route_request", return_value=_local_route(0.01)),
            patch.object(main, "_lookup_rdap", return_value=_rdap_ok()),
            patch.object(
                main,
                "_check_safe_browsing",
                return_value={
                    "source": "google_safe_browsing",
                    "checked": False,
                    "error": "SAFE_BROWSING_API_KEY is not configured.",
                },
            ),
            patch.object(main, "_retrieve_intel_articles", return_value=_knowledge_ok()),
        ):
            result = main.decide_url_threat("https://example.com")

        self.assertEqual(result["verdict"], "needs_review")
        self.assertLess(result["risk_score"], 0.5)
        self.assertTrue(any("Safe Browsing" in item for item in result["unknowns"]))


if __name__ == "__main__":
    unittest.main()
