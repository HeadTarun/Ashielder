from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def _email(
    subject: str = "Weekly receipt",
    sender: str = "Store <receipts@example.com>",
    body: str = "Thanks for your order.",
    urls: list[str] | None = None,
) -> dict:
    return {
        "id": "msg_001",
        "subject": subject,
        "sender": sender,
        "timestamp": "2026-06-17T12:00:00Z",
        "body": body,
        "urls": urls or [],
    }


def _classification(label: str = "LABEL_1") -> dict:
    return {
        "model": "distilbert_m2",
        "available": True,
        "label": label,
        "confidence": 0.95,
        "route_hint": "analysis",
    }


def _threat(spam_probability: float = 0.1) -> dict:
    return {
        "verdict": "analysis_ready",
        "risk_score": 0.1,
        "reason": "Request was routed to semantic analysis.",
        "evidence": {
            "route": "analysis",
            "result": {"model": "distilbert_m2", "spam_probability": spam_probability},
        },
        "recommended_actions": [],
        "memory_updated": True,
    }


def _embedding() -> dict:
    return {"model": "embed", "available": True, "embedding": [0.1, 0.2], "embedding_dim": 2}


def _url_decision(verdict: str = "low_risk", risk_score: float = 0.1) -> dict:
    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "confidence": 0.8,
        "evidence": {},
        "recommended_actions": [],
        "unknowns": [],
    }


class MailboxDiagnosisTests(unittest.TestCase):
    def test_risk_level_thresholds(self) -> None:
        self.assertEqual(main._risk_level(0), "SAFE")
        self.assertEqual(main._risk_level(24), "SAFE")
        self.assertEqual(main._risk_level(25), "LOW RISK")
        self.assertEqual(main._risk_level(50), "MEDIUM RISK")
        self.assertEqual(main._risk_level(75), "HIGH RISK")
        self.assertEqual(main._risk_level(90), "CRITICAL")

    def test_sender_analysis_flags_lookalike_domain(self) -> None:
        result = main._analyze_sender(
            "PayPal Support <security@paypa1.com>",
            "Verify your PayPal account",
            "Your PayPal account needs review.",
        )

        self.assertTrue(result["suspicious"])
        self.assertIn("lookalike_domain", result["issues"])
        self.assertTrue(result["evidence"])

    def test_sender_analysis_allows_normal_sender(self) -> None:
        result = main._analyze_sender(
            "Example Store <receipts@example.com>",
            "Your receipt",
            "Thanks for your order.",
        )

        self.assertFalse(result["suspicious"])
        self.assertEqual(result["evidence"], [])

    def test_email_report_requires_evidence_for_findings(self) -> None:
        with (
            patch.object(main, "classify_request", return_value=_classification()),
            patch.object(main, "analyze_threat", return_value=_threat()),
            patch.object(main, "analyze_embedding", return_value=_embedding()),
        ):
            report = main._analyze_email_security(_email())

        self.assertEqual(report["risk_level"], "SAFE")
        self.assertEqual(report["risk_score"], 0)
        self.assertEqual(report["evidence"], [])
        self.assertEqual(main._mailbox_findings([report]), [])

    def test_high_risk_email_aggregates_with_evidence(self) -> None:
        suspicious = _email(
            subject="Urgent PayPal account verification required",
            sender="PayPal Support <security@paypa1.com>",
            body="Verify your account password immediately at https://paypa1.com/login",
            urls=["https://paypa1.com/login"],
        )
        with (
            patch.object(main, "classify_request", return_value=_classification("LABEL_0")),
            patch.object(main, "analyze_threat", return_value=_threat(0.95)),
            patch.object(main, "analyze_embedding", return_value=_embedding()),
            patch.object(main, "decide_url_threat", return_value=_url_decision("high_risk", 0.97)),
        ):
            report = main._analyze_email_security(suspicious)

        self.assertEqual(report["risk_score"], 100)
        self.assertEqual(report["risk_level"], "CRITICAL")
        self.assertTrue(report["suspicious_urls"])
        self.assertGreaterEqual(len(report["evidence"]), 3)
        findings = main._mailbox_findings([report])
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0]["evidence"])

    def test_mailbox_aggregation_contract(self) -> None:
        emails = [
            _email(),
            _email(
                subject="Urgent Google account verification",
                sender="Google Support <alerts@goog1e.com>",
                body="Confirm your password immediately at https://goog1e.com/login",
                urls=["https://goog1e.com/login"],
            ),
        ]
        with (
            patch.object(main, "gmail_fetch_tool", return_value={"emails": emails}),
            patch.object(main, "classify_request", return_value=_classification()),
            patch.object(main, "analyze_threat", return_value=_threat()),
            patch.object(main, "analyze_embedding", return_value=_embedding()),
            patch.object(main, "decide_url_threat", return_value=_url_decision("high_risk", 0.97)),
        ):
            result = main.analyze_latest_gmail()

        self.assertIn(result["overall_status"], {"ATTENTION REQUIRED", "HIGH RISK", "CRITICAL"})
        self.assertEqual(result["summary"]["emails_analyzed"], 2)
        self.assertEqual(result["summary"]["high_risk_emails"], 1)
        self.assertTrue(result["findings"])
        self.assertIn("MAILBOX SECURITY DIAGNOSIS", result["mailbox_diagnosis"])
        self.assertIn("MAILGUARD AI SECURITY REPORT", result["system_report"])


if __name__ == "__main__":
    unittest.main()
