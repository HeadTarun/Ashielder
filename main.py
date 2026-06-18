from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime, parseaddr
import html as html_lib
import json
import logging
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import requests
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi.openapi.models import OAuth2, OAuthFlowAuthorizationCode, OAuthFlows
from google.adk.agents import Agent
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.tools import ToolContext
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

try:
    from google.adk.models.lite_llm import LiteLlm
except ImportError:  # pragma: no cover - handled by dependency sync
    LiteLlm = None


load_dotenv()

LOGGER = logging.getLogger(__name__)
URL_RE = re.compile(r"https?://[^\s<>'\"()]+|www\.[^\s<>'\"()]+", re.IGNORECASE)
DOMAIN_RE = re.compile(r"(?<!@)\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL_TOKEN_URI = "https://oauth2.googleapis.com/token"
GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1"
SUSPICIOUS_TLDS = {"zip", "mov", "click", "top", "xyz", "work", "support", "quest", "tk", "ml", "ga", "cf", "gq"}
TRUSTED_BRANDS = {
    "amazon": {"amazon.com"},
    "apple": {"apple.com"},
    "google": {"google.com", "gmail.com", "accounts.google.com"},
    "microsoft": {"microsoft.com", "live.com", "outlook.com"},
    "netflix": {"netflix.com"},
    "paypal": {"paypal.com"},
}
PHISHING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(password|passcode|one[-\s]?time code|otp|security code)\b",
        r"\b(verify|validate|confirm|restore|unlock|reactivate)\b.{0,80}\b(account|identity|login|payment|wallet)\b",
        r"\b(sign in|login|log in)\b.{0,80}\b(now|immediately|within|before)\b",
    ]
]
URGENCY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(urgent|immediate|action required|final notice|last warning)\b",
        r"\b(within 24 hours|today only|expires today|account suspended|account locked)\b",
    ]
]


def _split_csv(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return default
    items = {item.strip() for item in value.split(",")}
    return {item for item in items if item} or default


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _extract_url(text: str) -> str | None:
    match = URL_RE.search(text)
    if not match:
        return None
    url = match.group(0)
    if url.startswith("www."):
        return f"https://{url}"
    return url


@dataclass(frozen=True)
class RouteConfig:
    url_spam_labels: set[str]
    analysis_labels: set[str]
    min_confidence: float


class DistilBertClassifier:
    _MODEL_PATH_ENV = "DISTILBERT_MODEL_PATH"
    _MODEL_LOCK = threading.Lock()
    _MODEL_INITIALIZED = False
    _MODEL_AVAILABLE = False
    _TOKENIZER: Any = None
    _SESSION: Any = None
    _ID2LABEL: dict[str, str] = {}

    def __init__(self) -> None:
        self._ensure_model_runtime()

    @classmethod
    def _resolve_model_path(cls) -> Path:
        raw_path = os.getenv(cls._MODEL_PATH_ENV)
        if raw_path:
            return Path(raw_path).expanduser()
        return Path(__file__).resolve().parent / "distilbert_m2"

    @classmethod
    def _ensure_model_runtime(cls) -> None:
        if cls._MODEL_INITIALIZED:
            return

        with cls._MODEL_LOCK:
            if cls._MODEL_INITIALIZED:
                return

            try:
                model_path = cls._resolve_model_path()
                with (model_path / "config.json").open("r", encoding="utf-8") as config_file:
                    config = json.load(config_file)

                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=512)
                tokenizer.enable_padding(length=512, pad_id=0, pad_token="[PAD]")
                cls._TOKENIZER = tokenizer
                cls._SESSION = ort.InferenceSession(
                    str(model_path / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                cls._ID2LABEL = {str(key): value for key, value in config.get("id2label", {}).items()}
                cls._MODEL_AVAILABLE = True
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError):
                cls._TOKENIZER = None
                cls._SESSION = None
                cls._ID2LABEL = {}
                cls._MODEL_AVAILABLE = False
            finally:
                cls._MODEL_INITIALIZED = True

    def classify(self, text: str) -> dict[str, Any]:
        if not self.__class__._MODEL_AVAILABLE:
            return {
                "model": "distilbert_m2",
                "available": False,
                "label": None,
                "confidence": None,
                "route_hint": "general",
            }

        tokenizer = self.__class__._TOKENIZER
        session = self.__class__._SESSION
        if tokenizer is None or session is None:
            return {
                "model": "distilbert_m2",
                "available": False,
                "label": None,
                "confidence": None,
                "route_hint": "general",
            }

        encoding = tokenizer.encode(text)
        session_inputs = {}
        for input_def in session.get_inputs():
            if input_def.name == "input_ids":
                session_inputs[input_def.name] = np.asarray([encoding.ids], dtype=np.int64)
            elif input_def.name == "attention_mask":
                session_inputs[input_def.name] = np.asarray([encoding.attention_mask], dtype=np.int64)

        logits = np.asarray(session.run(None, session_inputs)[0], dtype=np.float32)[0]
        probabilities = _softmax(logits)
        ranking = np.argsort(-probabilities)
        top_index = int(ranking[0])
        label = self.__class__._ID2LABEL.get(str(top_index), f"LABEL_{top_index}")
        confidence = float(probabilities[top_index])
        route_hint = _route_from_label(label, confidence, text)
        score_map = {
            self.__class__._ID2LABEL.get(str(index), f"LABEL_{index}"): round(float(probabilities[index]), 6)
            for index in range(len(probabilities))
        }
        return {
            "model": "distilbert_m2",
            "available": True,
            "label": label,
            "confidence": round(confidence, 6),
            "route_hint": route_hint,
            "top_scores": score_map,
        }


class LightUrlSpamDetector:
    _MODEL_PATH_ENV = "LIGHTURLNET_MODEL_PATH"
    _MODEL_LOCK = threading.Lock()
    _MODEL_INITIALIZED = False
    _MODEL_AVAILABLE = False
    _SESSION: Any = None
    _CHAR_TO_IDX: dict[str, int] = {}
    _MAX_LEN = 256

    def __init__(self) -> None:
        self._ensure_model_runtime()

    @classmethod
    def _resolve_model_path(cls) -> Path:
        raw_path = os.getenv(cls._MODEL_PATH_ENV)
        if raw_path:
            return Path(raw_path).expanduser()
        return Path(__file__).resolve().parent / "lighturlnet"

    @classmethod
    def _ensure_model_runtime(cls) -> None:
        if cls._MODEL_INITIALIZED:
            return

        with cls._MODEL_LOCK:
            if cls._MODEL_INITIALIZED:
                return

            try:
                model_path = cls._resolve_model_path()
                with (model_path / "char2idx.json").open("r", encoding="utf-8") as vocab_file:
                    vocab = json.load(vocab_file)
                mapping = vocab.get("char_to_idx", vocab)
                cls._CHAR_TO_IDX = {str(key): int(value) for key, value in mapping.items()}
                cls._SESSION = ort.InferenceSession(
                    str(model_path / "lighturlnet.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                cls._MODEL_AVAILABLE = True
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError):
                cls._CHAR_TO_IDX = {}
                cls._SESSION = None
                cls._MODEL_AVAILABLE = False
            finally:
                cls._MODEL_INITIALIZED = True

    def predict(self, text: str) -> dict[str, Any]:
        if not self.__class__._MODEL_AVAILABLE or self.__class__._SESSION is None:
            return {
                "model": "lighturlnet",
                "available": False,
                "spam_probability": None,
                "spam": None,
                "input": text,
            }

        normalized = text.strip().lower()
        encoded = [self.__class__._CHAR_TO_IDX.get(char, 1) for char in normalized[: self.__class__._MAX_LEN]]
        if len(encoded) < self.__class__._MAX_LEN:
            encoded.extend([0] * (self.__class__._MAX_LEN - len(encoded)))

        input_ids = np.asarray([encoded], dtype=np.int64)
        logits = np.asarray(self.__class__._SESSION.run(None, {"input_ids": input_ids})[0], dtype=np.float32).reshape(-1)
        logit = float(logits[0])
        spam_probability = float(1.0 / (1.0 + math.exp(-logit)))
        return {
            "model": "lighturlnet",
            "available": True,
            "spam_probability": round(spam_probability, 6),
            "spam": spam_probability >= 0.5,
            "input": text,
        }


class EmbeddingAnalyzer:
    _MODEL_PATH_ENV = "EMBEDDING_MODEL_PATH"
    _MODEL_LOCK = threading.Lock()
    _MODEL_INITIALIZED = False
    _MODEL_AVAILABLE = False
    _TOKENIZER: Any = None
    _SESSION: Any = None
    _MAX_LEN = 256

    def __init__(self) -> None:
        self._ensure_model_runtime()

    @classmethod
    def _resolve_model_path(cls) -> Path:
        raw_path = os.getenv(cls._MODEL_PATH_ENV)
        if raw_path:
            return Path(raw_path).expanduser()
        return Path(__file__).resolve().parent / "embed"

    @classmethod
    def _ensure_model_runtime(cls) -> None:
        if cls._MODEL_INITIALIZED:
            return

        with cls._MODEL_LOCK:
            if cls._MODEL_INITIALIZED:
                return

            try:
                model_path = cls._resolve_model_path()
                with (model_path / "config.json").open("r", encoding="utf-8") as config_file:
                    config = json.load(config_file)

                from tokenizers import Tokenizer

                cls._MAX_LEN = int(config.get("max_len", 256))
                tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=cls._MAX_LEN)
                tokenizer.enable_padding(length=cls._MAX_LEN, pad_id=0, pad_token="[PAD]")
                cls._TOKENIZER = tokenizer
                cls._SESSION = ort.InferenceSession(
                    str(model_path / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                cls._MODEL_AVAILABLE = True
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError):
                cls._TOKENIZER = None
                cls._SESSION = None
                cls._MODEL_AVAILABLE = False
            finally:
                cls._MODEL_INITIALIZED = True

    def embed(self, text: str) -> dict[str, Any]:
        if not self.__class__._MODEL_AVAILABLE:
            return {
                "model": "embed",
                "available": False,
                "embedding": None,
                "embedding_dim": None,
            }

        tokenizer = self.__class__._TOKENIZER
        session = self.__class__._SESSION
        if tokenizer is None or session is None:
            return {
                "model": "embed",
                "available": False,
                "embedding": None,
                "embedding_dim": None,
            }

        encoding = tokenizer.encode(text)
        session_inputs = {
            input_def.name: np.asarray([encoding.ids], dtype=np.int64)
            for input_def in session.get_inputs()
            if input_def.name == "input_ids"
        }
        if any(input_def.name == "attention_mask" for input_def in session.get_inputs()):
            session_inputs["attention_mask"] = np.asarray([encoding.attention_mask], dtype=np.int64)
        if any(input_def.name == "token_type_ids" for input_def in session.get_inputs()):
            session_inputs["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)
        outputs = session.run(None, session_inputs)
        last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
        attention_mask = np.asarray([encoding.attention_mask], dtype=np.float32)[..., np.newaxis]
        masked_sum = (last_hidden_state * attention_mask).sum(axis=1)
        token_counts = np.maximum(attention_mask.sum(axis=1), 1.0)
        embedding = (masked_sum / token_counts).reshape(-1)
        vector = [round(float(value), 6) for value in embedding.tolist()]
        return {
            "model": "embed",
            "available": True,
            "embedding": vector,
            "embedding_dim": len(vector),
            "preview": vector[:8],
        }


_CLASSIFIER = DistilBertClassifier()
_URL_DETECTOR = LightUrlSpamDetector()
_EMBEDDER = EmbeddingAnalyzer()
SESSION_SERVICE = InMemorySessionService()
MEMORY_SERVICE = InMemoryMemoryService()
_LOCAL_DEMO_STATE: dict[str, Any] = {}
_INTEL_CORPUS_CACHE: list[dict[str, Any]] | None = None
_ROUTE_CONFIG = RouteConfig(
    url_spam_labels=_split_csv(os.getenv("DISTILBERT_URL_LABELS"), {"LABEL_0"}),
    analysis_labels=_split_csv(os.getenv("DISTILBERT_ANALYSIS_LABELS"), {"LABEL_1"}),
    min_confidence=float(os.getenv("DISTILBERT_MIN_CONFIDENCE", "0.5")),
)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _route_from_label(label: str | None, confidence: float, text: str) -> str:
    if label in _ROUTE_CONFIG.url_spam_labels:
        return "url_spam"
    if _extract_url(text):
        return "url_spam"
    return "analysis"


def classify_request(text: str) -> dict[str, Any]:
    """Classify text with DistilBERT and return the suggested route."""
    return _CLASSIFIER.classify(text)


def detect_url_spam(text: str) -> dict[str, Any]:
    """Score a URL or URL-like text with the LightURLNet spam model."""
    url = _extract_url(text) or text
    result = _URL_DETECTOR.predict(url)
    result["detected_url"] = url
    return result


def analyze_embedding(text: str) -> dict[str, Any]:
    """Generate an embedding for semantic analysis."""
    return _EMBEDDER.embed(text)


def _extract_domain(text: str) -> str | None:
    url = _extract_url(text) or text.strip()
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        return host
    match = DOMAIN_RE.search(text)
    return match.group(0).lower() if match else None


def _canonical_url(text: str) -> str | None:
    url = _extract_url(text)
    if url:
        return url
    domain = _extract_domain(text)
    return f"https://{domain}" if domain else None


def _event_date(events: list[dict[str, Any]], action: str) -> str | None:
    for event in events:
        if str(event.get("eventAction", "")).lower() == action:
            return event.get("eventDate")
    return None


def _domain_age_days(created_at: str | None) -> int | None:
    if not created_at:
        return None
    normalized = created_at.replace("Z", "+00:00")
    try:
        created = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - created).days, 0)


def _entity_name(entity: dict[str, Any]) -> str | None:
    vcard = entity.get("vcardArray")
    if not isinstance(vcard, list) or len(vcard) < 2:
        return None
    for item in vcard[1]:
        if isinstance(item, list) and len(item) >= 4 and item[0] == "fn":
            return str(item[3])
    return None


def _lookup_rdap(domain: str) -> dict[str, Any]:
    timeout = _safe_float_env("RDAP_TIMEOUT_SECONDS", 5.0)
    url = f"https://rdap.org/domain/{domain}"
    try:
        response = requests.get(url, timeout=timeout, headers={"Accept": "application/rdap+json, application/json"})
        if response.status_code == 404:
            return {
                "source": "rdap",
                "checked": True,
                "found": False,
                "domain": domain,
                "lookup_url": url,
            }
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {
            "source": "rdap",
            "checked": False,
            "domain": domain,
            "lookup_url": url,
            "error": str(exc),
        }

    events = data.get("events", []) if isinstance(data.get("events"), list) else []
    created_at = _event_date(events, "registration")
    expires_at = _event_date(events, "expiration")
    registrar = None
    for entity in data.get("entities", []) if isinstance(data.get("entities"), list) else []:
        roles = {str(role).lower() for role in entity.get("roles", [])}
        if "registrar" in roles:
            registrar = _entity_name(entity)
            break

    nameservers = []
    for nameserver in data.get("nameservers", []) if isinstance(data.get("nameservers"), list) else []:
        name = nameserver.get("ldhName") or nameserver.get("unicodeName")
        if name:
            nameservers.append(str(name).lower())

    return {
        "source": "rdap",
        "checked": True,
        "found": True,
        "domain": domain,
        "lookup_url": url,
        "registrar": registrar,
        "created_at": created_at,
        "expires_at": expires_at,
        "domain_age_days": _domain_age_days(created_at),
        "statuses": data.get("status", []),
        "nameservers": nameservers[:8],
        "secure_dns": data.get("secureDNS", {}),
    }


def _check_safe_browsing(url: str) -> dict[str, Any]:
    api_key = os.getenv("SAFE_BROWSING_API_KEY", "").strip()
    if not api_key:
        return {
            "source": "google_safe_browsing",
            "checked": False,
            "url": url,
            "error": "SAFE_BROWSING_API_KEY is not configured.",
        }

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    body = {
        "client": {
            "clientId": "tri-model-threat-agent",
            "clientVersion": "0.1.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        response = requests.post(endpoint, json=body, timeout=8)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {
            "source": "google_safe_browsing",
            "checked": False,
            "url": url,
            "error": str(exc),
        }

    matches = payload.get("matches", [])
    return {
        "source": "google_safe_browsing",
        "checked": True,
        "url": url,
        "matched": bool(matches),
        "matches": matches,
        "note": "No match found in checked sources." if not matches else "URL matched a Google Safe Browsing threat list.",
    }


def _default_corpus_path() -> Path:
    raw_path = os.getenv("INTEL_CORPUS_PATH")
    if raw_path:
        return Path(raw_path).expanduser()
    return Path(__file__).resolve().parent / "knowledge" / "intel_corpus.json"


def _load_intel_corpus() -> list[dict[str, Any]]:
    global _INTEL_CORPUS_CACHE
    if _INTEL_CORPUS_CACHE is not None:
        return _INTEL_CORPUS_CACHE

    path = _default_corpus_path()
    try:
        with path.open("r", encoding="utf-8") as corpus_file:
            payload = json.load(corpus_file)
    except (OSError, ValueError):
        _INTEL_CORPUS_CACHE = []
        return _INTEL_CORPUS_CACHE

    chunks = payload.get("chunks", payload if isinstance(payload, list) else [])
    prepared = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        enriched = dict(item)
        enriched["content"] = text
        enriched["embedding_text"] = " ".join(
            str(part)
            for part in [
                enriched.get("title", ""),
                enriched.get("content", ""),
                " ".join(enriched.get("keywords", [])),
            ]
            if part
        )
        prepared.append(enriched)
    _INTEL_CORPUS_CACHE = prepared
    return prepared


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denominator)


def _keyword_score(query: str, item: dict[str, Any]) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    haystack = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("content", "")),
            " ".join(item.get("keywords", [])),
        ]
    ).lower()
    if not query_terms:
        return 0.0
    return sum(1 for term in query_terms if term in haystack) / len(query_terms)


def _retrieve_intel_articles(query: str, top_k: int = 3) -> dict[str, Any]:
    corpus = _load_intel_corpus()
    if not corpus:
        return {
            "source": "local_intel_corpus",
            "checked": False,
            "matches": [],
            "error": "No local intelligence corpus was found.",
        }

    query_embedding_result = analyze_embedding(query)
    query_embedding = query_embedding_result.get("embedding")
    scored = []
    embedding_available = isinstance(query_embedding, list) and bool(query_embedding)
    for item in corpus:
        score = 0.0
        method = "keyword"
        if embedding_available:
            item_embedding = item.get("embedding")
            if not isinstance(item_embedding, list):
                item_embedding_result = analyze_embedding(item["embedding_text"])
                item_embedding = item_embedding_result.get("embedding")
                if isinstance(item_embedding, list):
                    item["embedding"] = item_embedding
            if isinstance(item_embedding, list):
                score = _cosine_similarity(query_embedding, item_embedding)
                method = "embedding"
        if score == 0.0:
            score = _keyword_score(query, item)
        scored.append((score, method, item))

    matches = []
    for score, method, item in sorted(scored, key=lambda entry: entry[0], reverse=True)[:top_k]:
        if score <= 0:
            continue
        matches.append(
            {
                "chunk_id": item.get("id"),
                "title": item.get("title"),
                "source": item.get("source", "local curated corpus"),
                "score": round(float(score), 6),
                "retrieval_method": method,
                "guidance": item.get("summary") or item.get("content", "")[:240],
                "signals": item.get("signals", []),
            }
        )

    return {
        "source": "local_intel_corpus",
        "checked": True,
        "embedding_model": query_embedding_result.get("model"),
        "embedding_available": embedding_available,
        "matches": matches,
    }


def _safe_browsing_risk(safe_browsing: dict[str, Any]) -> tuple[list[str], list[str], float | None]:
    signals: list[str] = []
    unknowns: list[str] = []
    if not safe_browsing.get("checked"):
        unknowns.append(f"Google Safe Browsing was not checked: {safe_browsing.get('error', 'unknown error')}")
        return signals, unknowns, None
    if safe_browsing.get("matched"):
        threat_types = sorted({str(match.get("threatType", "UNKNOWN")) for match in safe_browsing.get("matches", [])})
        signals.append(f"Google Safe Browsing matched threat list(s): {', '.join(threat_types)}.")
        return signals, unknowns, 0.95
    return signals, unknowns, 0.05


def _rdap_risk(rdap: dict[str, Any]) -> tuple[list[str], list[str], float | None]:
    signals: list[str] = []
    unknowns: list[str] = []
    if not rdap.get("checked"):
        unknowns.append(f"RDAP was not checked: {rdap.get('error', 'unknown error')}")
        return signals, unknowns, None
    if rdap.get("found") is False:
        signals.append("RDAP did not find registration data for the domain.")
        return signals, unknowns, 0.55

    age = rdap.get("domain_age_days")
    if isinstance(age, int):
        if age < 7:
            signals.append("RDAP shows the domain was registered less than 7 days ago.")
            return signals, unknowns, 0.75
        if age < 30:
            signals.append("RDAP shows the domain was registered less than 30 days ago.")
            return signals, unknowns, 0.55
    else:
        unknowns.append("RDAP did not expose a parseable registration date.")

    statuses = {str(status).lower() for status in rdap.get("statuses", [])}
    hold_statuses = {status for status in statuses if "hold" in status}
    if hold_statuses:
        signals.append(f"RDAP status needs review: {', '.join(sorted(hold_statuses))}.")
        return signals, unknowns, 0.35
    return signals, unknowns, 0.1


def _local_ml_risk(route_result: dict[str, Any]) -> tuple[list[str], list[str], float]:
    signals: list[str] = []
    unknowns: list[str] = []
    classification = route_result.get("classification", {})
    result = route_result.get("result", {})
    confidence = classification.get("confidence")

    if classification.get("available") is False:
        unknowns.append("DistilBERT classifier was unavailable.")
    if result.get("available") is False:
        unknowns.append(f"{result.get('model', 'Downstream local model')} was unavailable.")

    spam_probability = result.get("spam_probability")
    if isinstance(spam_probability, float):
        if spam_probability >= 0.7:
            signals.append(f"LightURLNet spam probability is high ({spam_probability}).")
            return signals, unknowns, spam_probability
        if spam_probability >= 0.5:
            signals.append(f"LightURLNet spam probability needs review ({spam_probability}).")
            return signals, unknowns, spam_probability
        return signals, unknowns, spam_probability

    if isinstance(confidence, float) and confidence < _ROUTE_CONFIG.min_confidence:
        signals.append(f"Local classifier confidence is below threshold ({confidence}).")
        return signals, unknowns, 0.45
    return signals, unknowns, float(confidence or 0.1)


def _decision_from_evidence(
    route_result: dict[str, Any],
    rdap: dict[str, Any],
    safe_browsing: dict[str, Any],
    knowledge: dict[str, Any],
) -> dict[str, Any]:
    signals: list[str] = []
    unknowns: list[str] = []
    risk_scores: list[float] = []

    local_signals, local_unknowns, local_score = _local_ml_risk(route_result)
    rdap_signals, rdap_unknowns, rdap_score = _rdap_risk(rdap)
    safe_signals, safe_unknowns, safe_score = _safe_browsing_risk(safe_browsing)
    signals.extend(local_signals + rdap_signals + safe_signals)
    unknowns.extend(local_unknowns + rdap_unknowns + safe_unknowns)
    risk_scores.append(local_score)
    if rdap_score is not None:
        risk_scores.append(rdap_score)
    if safe_score is not None:
        risk_scores.append(safe_score)

    decision_signals = list(signals)
    corpus_matches = knowledge.get("matches", [])
    if knowledge.get("checked") is False:
        unknowns.append(str(knowledge.get("error", "Local intelligence corpus was unavailable.")))
    for match in corpus_matches[:2]:
        for signal in match.get("signals", [])[:2]:
            signals.append(f"Retrieved guidance {match.get('chunk_id')} notes: {signal}.")

    score = max(risk_scores) if risk_scores else 0.5
    strong_signal_count = sum(
        1
        for item in decision_signals
        if "high" in item.lower() or "matched" in item.lower() or "less than 7" in item.lower()
    )
    review_signal_count = len(decision_signals)

    if safe_browsing.get("matched") or score >= 0.7 or strong_signal_count >= 1 and review_signal_count >= 2:
        verdict = "high_risk"
        confidence = 0.9 if safe_browsing.get("matched") else 0.78
    elif unknowns or score >= 0.45 or review_signal_count:
        verdict = "needs_review"
        confidence = 0.65 if unknowns else 0.7
    else:
        verdict = "low_risk"
        confidence = 0.72

    return {
        "verdict": verdict,
        "risk_score": round(float(min(max(score, 0.0), 1.0)), 6),
        "confidence": round(confidence, 6),
        "signals": signals,
        "unknowns": unknowns,
    }


def _intel_recommendations(verdict: str) -> list[str]:
    if verdict == "high_risk":
        return [
            "Do not open the URL.",
            "Quarantine or report the message.",
            "Use the evidence object before overriding this verdict.",
        ]
    if verdict == "needs_review":
        return [
            "Ask a human reviewer to inspect the URL.",
            "Re-run with Safe Browsing configured if it was unavailable.",
            "Use RDAP and local ML evidence together; do not treat one missing source as safe.",
        ]
    return [
        "No checked source produced a strong threat signal.",
        "Proceed with normal caution and keep the evidence for comparison.",
    ]


def decide_url_threat(text_or_url: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Make an evidence-grounded URL threat decision using local ML, RDAP, Safe Browsing, and curated intel."""
    url = _canonical_url(text_or_url)
    domain = _extract_domain(text_or_url)
    if not url or not domain:
        return {
            "verdict": "needs_review",
            "risk_score": 0.5,
            "confidence": 0.4,
            "evidence": {},
            "citations": [],
            "recommended_actions": ["Provide a valid URL or domain and re-run the decision tool."],
            "unknowns": ["No URL or domain could be extracted from the input."],
        }

    route_result = route_request(url)
    rdap = _lookup_rdap(domain)
    safe_browsing = _check_safe_browsing(url)
    knowledge_query = f"{text_or_url} domain age rdap safe browsing phishing url risk"
    knowledge = _retrieve_intel_articles(knowledge_query)
    decision = _decision_from_evidence(route_result, rdap, safe_browsing, knowledge)

    citations = [
        {
            "source": "local_ml",
            "detail": "DistilBERT route classifier and LightURLNet URL model output.",
        },
        {
            "source": "rdap",
            "detail": rdap.get("lookup_url"),
        },
        {
            "source": "google_safe_browsing",
            "detail": "Google Safe Browsing threatMatches.find lookup."
            if safe_browsing.get("checked")
            else safe_browsing.get("error"),
        },
    ]
    citations.extend(
        {
            "source": "local_intel_corpus",
            "chunk_id": match.get("chunk_id"),
            "title": match.get("title"),
        }
        for match in knowledge.get("matches", [])
    )

    result = {
        "verdict": decision["verdict"],
        "risk_score": decision["risk_score"],
        "confidence": decision["confidence"],
        "evidence": {
            "input": text_or_url,
            "canonical_url": url,
            "domain": domain,
            "local_ml": route_result,
            "rdap": rdap,
            "safe_browsing": safe_browsing,
            "knowledge": knowledge,
            "signals": decision["signals"],
        },
        "citations": citations,
        "recommended_actions": _intel_recommendations(decision["verdict"]),
        "unknowns": decision["unknowns"],
    }

    state = _state_from_context(tool_context)
    _append_state_list(
        state,
        "url_threat_decisions",
        {
            "url": url,
            "domain": domain,
            "verdict": result["verdict"],
            "risk_score": result["risk_score"],
        },
    )
    state["last_url_threat_decision"] = result
    return result


def _state_from_context(tool_context: ToolContext | None) -> dict[str, Any]:
    if tool_context is None:
        return _LOCAL_DEMO_STATE
    return tool_context.state


def _append_state_list(state: dict[str, Any], key: str, value: dict[str, Any], max_items: int = 8) -> None:
    items = list(state.get(key, []))
    items.append(value)
    state[key] = items[-max_items:]


def _risk_from_route(route_result: dict[str, Any]) -> dict[str, Any]:
    route = route_result.get("route")
    classification = route_result.get("classification", {})
    result = route_result.get("result", {})
    confidence = classification.get("confidence")

    if route == "url_spam":
        spam_probability = result.get("spam_probability")
        if result.get("spam") is True or (isinstance(spam_probability, float) and spam_probability >= 0.7):
            return {
                "verdict": "high_risk",
                "risk_score": round(float(spam_probability or confidence or 0.75), 6),
                "reason": "URL model marked the link as likely spam.",
            }
        return {
            "verdict": "low_risk",
            "risk_score": round(float(spam_probability or 0.1), 6),
            "reason": "URL model did not find strong spam evidence.",
        }

    if isinstance(confidence, float) and confidence < _ROUTE_CONFIG.min_confidence:
        return {
            "verdict": "needs_review",
            "risk_score": round(1.0 - confidence, 6),
            "reason": "Classifier confidence is below the review threshold.",
        }

    return {
        "verdict": "analysis_ready",
        "risk_score": round(float(confidence or 0.0), 6),
        "reason": "Request was routed to semantic analysis.",
    }


def route_request(text: str) -> dict[str, Any]:
    """Route a request across the three local models."""
    classification = classify_request(text)
    route = classification.get("route_hint", "general")
    result: dict[str, Any]

    if route == "url_spam":
        result = detect_url_spam(text)
    elif route == "analysis":
        result = analyze_embedding(text)
    else:
        result = {
            "model": "distilbert_m2",
            "available": classification.get("available", False),
            "label": classification.get("label"),
            "confidence": classification.get("confidence"),
            "note": "No specialized downstream tool selected.",
        }

    return {
        "route": route,
        "classification": classification,
        "result": result,
    }


def _analyze_threat_with_state(text: str, state: dict[str, Any]) -> dict[str, Any]:
    routed = route_request(text)
    risk = _risk_from_route(routed)
    memory_item = {
        "text": text[:240],
        "route": routed.get("route"),
        "verdict": risk["verdict"],
        "risk_score": risk["risk_score"],
        "model": routed.get("result", {}).get("model"),
    }
    state["last_analysis"] = memory_item
    _append_state_list(state, "analysis_history", memory_item)

    if routed.get("route") == "url_spam":
        url = routed.get("result", {}).get("detected_url")
        if url and risk["verdict"] in {"high_risk", "needs_review"}:
            blocked = list(state.get("user:flagged_urls", []))
            if url not in blocked:
                blocked.append(url)
            state["user:flagged_urls"] = blocked[-20:]

    return {
        "verdict": risk["verdict"],
        "risk_score": risk["risk_score"],
        "reason": risk["reason"],
        "evidence": routed,
        "recommended_actions": _recommended_actions(risk["verdict"]),
        "memory_updated": True,
    }


def analyze_threat(text: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Run the full local ML pipeline, produce a verdict, and store it in ADK session state."""
    return _analyze_threat_with_state(text, _state_from_context(tool_context))


def remember_user_fact(key: str, value: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Remember a small user preference or demo fact in ADK user-scoped state."""
    safe_key = re.sub(r"[^a-zA-Z0-9_:-]+", "_", key.strip().lower()).strip("_")
    if not safe_key:
        return {"status": "error", "message": "key is required"}
    state = _state_from_context(tool_context)
    state_key = safe_key if safe_key.startswith("user:") else f"user:{safe_key}"
    state[state_key] = value
    return {"status": "remembered", "key": state_key, "value": value}


def search_session_memory(query: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Search recent ADK session-state analysis history and remembered user facts."""
    state = _state_from_context(tool_context)
    query_lower = query.lower()
    history = state.get("analysis_history", [])
    facts = {key: value for key, value in state.items() if str(key).startswith("user:")}
    matched_history = [
        item
        for item in history
        if query_lower in json.dumps(item, ensure_ascii=True).lower()
    ]
    matched_facts = {
        key: value
        for key, value in facts.items()
        if query_lower in key.lower() or query_lower in str(value).lower()
    }
    return {
        "matches": matched_history,
        "remembered_facts": matched_facts,
        "last_analysis": state.get("last_analysis"),
        "history_count": len(history),
    }


def _gmail_user_key(tool_context: ToolContext | None) -> str:
    if tool_context is None:
        return "local"
    for attr in ("user_id", "session_id", "invocation_id"):
        value = getattr(tool_context, attr, None)
        if value:
            return str(value)
    return "adk_user"


def _gmail_token_db_path() -> Path:
    raw_path = os.getenv("GMAIL_TOKEN_DB_PATH", "data/gmail_tokens.db")
    return Path(raw_path).expanduser()


def _gmail_fetch_limit() -> int:
    try:
        return max(1, min(int(os.getenv("GMAIL_FETCH_LIMIT", "10")), 25))
    except ValueError:
        return 10


def _gmail_fernet() -> Fernet | None:
    raw_key = os.getenv("GMAIL_TOKEN_ENCRYPTION_KEY", "").strip()
    if not raw_key:
        return None
    try:
        return Fernet(raw_key.encode("utf-8"))
    except (ValueError, TypeError):
        digest = base64.urlsafe_b64encode(__import__("hashlib").sha256(raw_key.encode("utf-8")).digest())
        return Fernet(digest)


def _ensure_gmail_token_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_tokens (
                user_key TEXT PRIMARY KEY,
                encrypted_token BLOB NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _store_gmail_token(user_key: str, token_payload: dict[str, Any]) -> None:
    fernet = _gmail_fernet()
    if fernet is None:
        LOGGER.warning("GMAIL_TOKEN_ENCRYPTION_KEY is not configured; Gmail token was not persisted.")
        return
    db_path = _gmail_token_db_path()
    _ensure_gmail_token_table(db_path)
    encrypted = fernet.encrypt(json.dumps(token_payload, ensure_ascii=True).encode("utf-8"))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO gmail_tokens (user_key, encrypted_token, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_key) DO UPDATE SET
                encrypted_token = excluded.encrypted_token,
                updated_at = excluded.updated_at
            """,
            (user_key, encrypted, datetime.now(timezone.utc).isoformat()),
        )


def _load_gmail_token(user_key: str) -> dict[str, Any] | None:
    fernet = _gmail_fernet()
    if fernet is None:
        return None
    db_path = _gmail_token_db_path()
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT encrypted_token FROM gmail_tokens WHERE user_key = ?",
                (user_key,),
            ).fetchone()
        if not row:
            return None
        return json.loads(fernet.decrypt(row[0]).decode("utf-8"))
    except (sqlite3.Error, InvalidToken, ValueError, TypeError) as exc:
        LOGGER.warning("Failed to load Gmail token: %s", exc)
        return None


def _delete_gmail_token(user_key: str) -> None:
    db_path = _gmail_token_db_path()
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM gmail_tokens WHERE user_key = ?", (user_key,))
    except sqlite3.Error as exc:
        LOGGER.warning("Failed to delete Gmail token: %s", exc)


def _credential_oauth_payload(credential: Any) -> dict[str, Any]:
    if credential is None:
        return {}
    if isinstance(credential, dict):
        oauth = credential.get("oauth2") or credential.get("oauth") or credential
    else:
        oauth = getattr(credential, "oauth2", None) or credential
    if isinstance(oauth, dict):
        getter = oauth.get
    else:
        getter = lambda key, default=None: getattr(oauth, key, default)
    return {
        "token": getter("accessToken") or getter("access_token"),
        "refresh_token": getter("refreshToken") or getter("refresh_token"),
        "token_uri": getter("tokenUri") or getter("token_uri") or GMAIL_TOKEN_URI,
        "client_id": getter("clientId") or getter("client_id") or os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
        "client_secret": getter("clientSecret") or getter("client_secret") or os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
        "scopes": [GMAIL_READONLY_SCOPE],
        "expiry": _expiry_from_credential(getter("expiresAt") or getter("expiry")),
    }


def _expiry_from_credential(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        timestamp = int(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp // 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _credentials_from_payload(payload: dict[str, Any]) -> Credentials | None:
    token = payload.get("token")
    refresh_token = payload.get("refresh_token")
    client_id = payload.get("client_id") or os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = payload.get("client_secret") or os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    if not token and not refresh_token:
        return None
    expiry = None
    if payload.get("expiry"):
        try:
            expiry = datetime.fromisoformat(str(payload["expiry"]).replace("Z", "+00:00"))
        except ValueError:
            expiry = None
    return Credentials(
        token=token,
        refresh_token=refresh_token,
        token_uri=payload.get("token_uri") or GMAIL_TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=payload.get("scopes") or [GMAIL_READONLY_SCOPE],
        expiry=expiry,
    )


def _payload_from_credentials(credentials: Credentials) -> dict[str, Any]:
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or [GMAIL_READONLY_SCOPE]),
        "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
    }


def _gmail_credentials(credential: Any, tool_context: ToolContext | None) -> tuple[Credentials | None, dict[str, Any] | None]:
    user_key = _gmail_user_key(tool_context)
    payload = _credential_oauth_payload(credential) if credential is not None else {}
    if payload.get("token") or payload.get("refresh_token"):
        _store_gmail_token(user_key, payload)
    else:
        payload = _load_gmail_token(user_key) or {}

    credentials = _credentials_from_payload(payload)
    if credentials is None:
        return None, {
            "status": "auth_required",
            "message": "Connect Gmail with Google OAuth before running mailbox diagnosis.",
            "scope": GMAIL_READONLY_SCOPE,
        }

    try:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            _store_gmail_token(user_key, _payload_from_credentials(credentials))
        if not credentials.valid:
            return None, {
                "status": "auth_required",
                "message": "Gmail credentials are expired or incomplete. Reconnect Gmail.",
                "scope": GMAIL_READONLY_SCOPE,
            }
    except RefreshError as exc:
        _delete_gmail_token(user_key)
        return None, {
            "status": "auth_required",
            "message": "Gmail authorization was revoked or expired. Reconnect Gmail.",
            "error": str(exc),
            "scope": GMAIL_READONLY_SCOPE,
        }
    return credentials, None


def _gmail_auth_config() -> AuthConfig:
    scopes = {GMAIL_READONLY_SCOPE: "Read Gmail messages for security diagnosis."}
    return AuthConfig(
        authScheme=OAuth2(
            flows=OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl=GMAIL_AUTH_URI,
                    tokenUrl=GMAIL_TOKEN_URI,
                    scopes=scopes,
                )
            )
        ),
        rawAuthCredential=AuthCredential(
            authType=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                clientId=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
                clientSecret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
                redirectUri=os.getenv("GOOGLE_OAUTH_REDIRECT_URI"),
                tokenEndpointAuthMethod="client_secret_post",
            ),
        ),
        credentialKey="gmail_readonly",
    )


def _decode_gmail_body(data: str | None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")
    except (ValueError, OSError):
        return ""


def _strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _extract_mime_text(payload: dict[str, Any]) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime_type = str(part.get("mimeType", "")).lower()
        body_data = (part.get("body") or {}).get("data")
        if body_data and mime_type == "text/plain":
            plain_parts.append(_decode_gmail_body(body_data))
        elif body_data and mime_type == "text/html":
            html_parts.append(_strip_html(_decode_gmail_body(body_data)))
        for child in part.get("parts", []) or []:
            if isinstance(child, dict):
                walk(child)

    walk(payload or {})
    text = "\n".join(part for part in plain_parts if part.strip()) or "\n".join(part for part in html_parts if part.strip())
    return re.sub(r"\s+\n", "\n", text).strip()


def _gmail_headers(payload: dict[str, Any]) -> dict[str, str]:
    headers = {}
    for item in payload.get("headers", []) or []:
        name = str(item.get("name", "")).lower()
        value = str(item.get("value", ""))
        if name:
            headers[name] = value
    return headers


def _gmail_timestamp(message: dict[str, Any], headers: dict[str, str]) -> str:
    internal_date = message.get("internalDate")
    if internal_date:
        try:
            return datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError):
            pass
    date_header = headers.get("date")
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, IndexError, OverflowError):
            pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)\"]}'")
        if url.startswith("www."):
            url = f"https://{url}"
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def gmail_fetch_tool(credential: Any = None, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Fetch the latest Gmail messages as clean structured JSON."""
    credentials, error = _gmail_credentials(credential, tool_context)
    if error:
        return error

    session = AuthorizedSession(credentials)
    limit = _gmail_fetch_limit()
    try:
        list_response = session.get(
            f"{GMAIL_API_ROOT}/users/me/messages",
            params={"maxResults": limit, "q": "newer_than:30d"},
            timeout=15,
        )
        list_response.raise_for_status()
        message_refs = list_response.json().get("messages", []) or []
    except (requests.RequestException, ValueError) as exc:
        return {"status": "error", "message": "Failed to list Gmail messages.", "error": str(exc)}

    emails: list[dict[str, Any]] = []
    for ref in message_refs[:limit]:
        message_id = ref.get("id")
        if not message_id:
            continue
        try:
            message_response = session.get(
                f"{GMAIL_API_ROOT}/users/me/messages/{message_id}",
                params={"format": "full"},
                timeout=15,
            )
            message_response.raise_for_status()
            message = message_response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Failed to fetch Gmail message %s: %s", message_id, exc)
            continue

        payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
        headers = _gmail_headers(payload)
        body = _extract_mime_text(payload)
        combined_text = " ".join([headers.get("subject", ""), body])
        emails.append(
            {
                "id": message_id,
                "subject": headers.get("subject", ""),
                "sender": headers.get("from", ""),
                "timestamp": _gmail_timestamp(message, headers),
                "body": body,
                "urls": _extract_urls(combined_text),
            }
        )

    return {"emails": emails}


def _registered_domain(domain: str) -> str:
    parts = [part for part in domain.lower().strip(".").split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain.lower().strip(".")


def _domain_skeleton(value: str) -> str:
    translation = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "@": "a", "$": "s"})
    return value.lower().translate(translation).replace("rn", "m").replace("vv", "w")


def _is_lookalike_domain(domain: str, brand: str, trusted_domains: set[str]) -> bool:
    registered = _registered_domain(domain)
    if any(registered == trusted or registered.endswith(f".{trusted}") for trusted in trusted_domains):
        return False
    stem = registered.split(".", 1)[0]
    skeleton = _domain_skeleton(stem)
    return skeleton == brand or _levenshtein_distance(skeleton, brand) == 1


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def _analyze_sender(sender: str, subject: str = "", body: str = "") -> dict[str, Any]:
    display_name, address = parseaddr(sender)
    domain = address.split("@", 1)[-1].lower() if "@" in address else ""
    registered = _registered_domain(domain) if domain else ""
    evidence: list[str] = []
    issues: list[str] = []
    text = " ".join([display_name, subject, body[:500]]).lower()

    if domain:
        tld = domain.rsplit(".", 1)[-1]
        if tld in SUSPICIOUS_TLDS:
            issues.append("suspicious_tld")
            evidence.append(f"Sender domain uses suspicious TLD: .{tld}")

    for brand, trusted_domains in TRUSTED_BRANDS.items():
        if brand in text and domain and not any(registered == trusted for trusted in trusted_domains):
            issues.append("domain_mismatch")
            evidence.append(f"Message references {brand.title()} but sender domain is {registered or domain}.")
        if domain and _is_lookalike_domain(domain, brand, trusted_domains):
            issues.append("lookalike_domain")
            evidence.append(f"Sender domain {registered or domain} resembles trusted brand {brand}.")

    return {
        "display_name": display_name,
        "address": address,
        "domain": domain,
        "registered_domain": registered,
        "issues": sorted(set(issues)),
        "evidence": evidence,
        "suspicious": bool(evidence),
    }


def _risk_level(score: int) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH RISK"
    if score >= 50:
        return "MEDIUM RISK"
    if score >= 25:
        return "LOW RISK"
    return "SAFE"


def _finding_severity(score: int) -> str:
    level = _risk_level(score)
    return "High" if level == "HIGH RISK" else "Critical" if level == "CRITICAL" else "Medium" if level == "MEDIUM RISK" else "Low"


def _email_text(email: dict[str, Any]) -> str:
    return "\n".join(
        str(email.get(key, ""))
        for key in ("subject", "sender", "body")
        if email.get(key)
    )


def _phishing_evidence(text: str) -> list[str]:
    return [f"Phishing or credential language matched: {pattern.pattern}" for pattern in PHISHING_PATTERNS if pattern.search(text)]


def _urgency_evidence(text: str) -> list[str]:
    return [f"Urgency or account-pressure language matched: {pattern.pattern}" for pattern in URGENCY_PATTERNS if pattern.search(text)]


def _spam_probability_from_analysis(analysis: dict[str, Any]) -> float | None:
    result = analysis.get("evidence", {}).get("result", {})
    probability = result.get("spam_probability")
    if isinstance(probability, (float, int)):
        return float(probability)
    return None


def _analyze_email_security(email: dict[str, Any], tool_context: ToolContext | None = None) -> dict[str, Any]:
    text = _email_text(email)
    classification = classify_request(text)
    threat = analyze_threat(text, tool_context)
    embedding = analyze_embedding(text)
    sender = _analyze_sender(str(email.get("sender", "")), str(email.get("subject", "")), str(email.get("body", "")))
    evidence: list[str] = []
    suspicious_urls: list[str] = []
    score = 0

    url_decisions = []
    for url in email.get("urls", []) or []:
        decision = decide_url_threat(str(url), tool_context)
        url_decisions.append(decision)
        verdict = str(decision.get("verdict", ""))
        url_score = decision.get("risk_score")
        if verdict == "high_risk" or (isinstance(url_score, (float, int)) and float(url_score) >= 0.75):
            score += 60
            suspicious_urls.append(str(url))
            evidence.append(f"URL classified as high risk: {url}")
        elif verdict == "needs_review" or (isinstance(url_score, (float, int)) and float(url_score) >= 0.45):
            score += 30
            suspicious_urls.append(str(url))
            evidence.append(f"URL requires review: {url}")

    if sender["suspicious"]:
        score += 25
        evidence.extend(sender["evidence"])
    if "lookalike_domain" in sender["issues"]:
        score += 25
    phishing_matches = _phishing_evidence(text)
    if phishing_matches:
        score += 20
        evidence.extend(phishing_matches)
    urgency_matches = _urgency_evidence(text)
    if urgency_matches:
        score += 15
        evidence.extend(urgency_matches)
    spam_probability = _spam_probability_from_analysis(threat)
    if spam_probability is not None and spam_probability > 0.9:
        score += 10
        evidence.append(f"URL spam confidence is {spam_probability:.2f}.")

    score = min(score, 100)
    category = str(classification.get("label") or classification.get("route_hint") or "unknown")
    risk_level = _risk_level(score)
    action_items = _email_action_items(risk_level, bool(suspicious_urls), bool(sender["evidence"]))
    report = {
        "id": email.get("id", ""),
        "subject": email.get("subject", ""),
        "sender": email.get("sender", ""),
        "timestamp": email.get("timestamp", ""),
        "category": category,
        "risk_level": risk_level,
        "risk_score": score,
        "suspicious_urls": suspicious_urls,
        "evidence": evidence,
        "action_items": action_items,
        "model_evidence": {
            "classification": classification,
            "threat": threat,
            "embedding": {
                "model": embedding.get("model"),
                "available": embedding.get("available"),
                "embedding_dim": embedding.get("embedding_dim"),
            },
            "url_decisions": url_decisions,
            "sender": sender,
        },
    }
    return report


def _email_action_items(risk_level: str, has_urls: bool, has_sender_issue: bool) -> list[str]:
    if risk_level in {"CRITICAL", "HIGH RISK"}:
        actions = ["Do not click links or download attachments.", "Report or quarantine this email."]
        if has_sender_issue:
            actions.append("Verify the sender through a trusted channel.")
        return actions
    if risk_level in {"MEDIUM RISK", "LOW RISK"}:
        actions = ["Review this email before taking action."]
        if has_urls:
            actions.append("Open links only after confirming the destination.")
        return actions
    return ["No immediate action required."]


def _mailbox_recommendations(reports: list[dict[str, Any]]) -> list[str]:
    if any(report["risk_level"] in {"CRITICAL", "HIGH RISK"} for report in reports):
        return [
            "Avoid clicking suspicious links.",
            "Report or delete high-risk messages.",
            "Enable or verify two-factor authentication.",
            "Review recent Gmail account activity.",
        ]
    if any(report["risk_level"] in {"MEDIUM RISK", "LOW RISK"} for report in reports):
        return [
            "Review suspicious messages before clicking links.",
            "Verify unexpected account or payment requests directly with the sender.",
            "Keep two-factor authentication enabled.",
        ]
    return [
        "Continue normal mailbox hygiene.",
        "Keep two-factor authentication enabled.",
        "Report unexpected suspicious messages if they appear.",
    ]


def _overall_status(score: int) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH RISK"
    if score >= 25:
        return "ATTENTION REQUIRED"
    return "SAFE"


def _mailbox_findings(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for report in reports:
        if not report["evidence"]:
            continue
        if report["risk_score"] < 25:
            continue
        findings.append(
            {
                "severity": _finding_severity(report["risk_score"]),
                "issue": _issue_from_report(report),
                "email_id": report["id"],
                "subject": report["subject"],
                "evidence": report["evidence"],
            }
        )
    return findings


def _issue_from_report(report: dict[str, Any]) -> str:
    evidence_text = " ".join(report.get("evidence", [])).lower()
    if "url classified as high risk" in evidence_text:
        return "Potential phishing or malicious URL detected"
    if "resembles trusted brand" in evidence_text or "sender domain" in evidence_text:
        return "Sender trust issue detected"
    if "credential" in evidence_text or "phishing" in evidence_text:
        return "Potential credential harvesting attempt"
    return "Suspicious email indicators detected"


def _mailbox_diagnosis(summary: dict[str, int], score: int, reports: list[dict[str, Any]]) -> str:
    risk_level = _risk_level(score)
    suspicious = summary["suspicious_emails"]
    high = summary["high_risk_emails"]
    safe = summary["safe_emails"]
    primary = next((report for report in sorted(reports, key=lambda item: item["risk_score"], reverse=True) if report["evidence"]), None)
    lines = [
        "MAILBOX SECURITY DIAGNOSIS",
        "",
        f"Overall Security Status: {_overall_status(score)}",
        f"We analyzed {summary['emails_analyzed']} recent emails.",
        f"{safe} emails appear legitimate.",
        f"{suspicious} emails contain suspicious content.",
        f"{high} emails are high risk or critical.",
        f"Current Security Score: {score}/100",
        f"Risk Level: {risk_level}",
    ]
    if primary:
        lines.extend(
            [
                "",
                f"Primary Concern: {primary['subject'] or primary['id']}",
                f"This email was flagged because: {'; '.join(primary['evidence'][:3])}.",
            ]
        )
    lines.extend(["", "Recommended Actions:"])
    for index, action in enumerate(_mailbox_recommendations(reports), start=1):
        lines.append(f"{index}. {action}")
    return "\n".join(lines)


def _system_report(summary: dict[str, int], score: int, reports: list[dict[str, Any]]) -> str:
    high_or_critical = summary["high_risk_emails"]
    suspicious_urls = sum(len(report.get("suspicious_urls", [])) for report in reports)
    sender_issues = sum(1 for report in reports if report.get("model_evidence", {}).get("sender", {}).get("suspicious"))
    phishing = sum(1 for report in reports if any("Phishing" in evidence or "credential" in evidence for evidence in report.get("evidence", [])))
    primary = next((report for report in sorted(reports, key=lambda item: item["risk_score"], reverse=True) if report["evidence"]), None)
    lines = [
        "=================================",
        "MAILGUARD AI SECURITY REPORT",
        "=================================",
        "",
        f"Mailbox Status      : {_overall_status(score)}",
        f"Security Score      : {score}/100",
        f"Risk Level          : {_risk_level(score)}",
        "",
        f"Emails Analyzed     : {summary['emails_analyzed']}",
        f"Safe Emails         : {summary['safe_emails']}",
        f"Suspicious Emails   : {summary['suspicious_emails']}",
        f"Critical Emails     : {sum(1 for report in reports if report['risk_level'] == 'CRITICAL')}",
        "",
        f"URL Scan Status     : {'WARNING' if suspicious_urls else 'GOOD'}",
        f"Sender Trust Status : {'WARNING' if sender_issues else 'GOOD'}",
        f"Spam Activity       : {'MODERATE' if summary['suspicious_emails'] else 'LOW'}",
        f"Phishing Risk       : {'DETECTED' if phishing else 'NOT DETECTED'}",
        "",
        "PRIMARY CONCERN",
        "",
        primary["evidence"][0] if primary else "No evidence-backed high-risk concern was detected.",
        "",
        "FINAL DIAGNOSIS",
        "",
        _final_diagnosis_sentence(score, high_or_critical),
    ]
    return "\n".join(lines)


def _final_diagnosis_sentence(score: int, high_or_critical: int) -> str:
    if score >= 90:
        return "Your mailbox contains critical indicators that require immediate attention."
    if score >= 75:
        return "Your mailbox has high-risk messages that should be reported or removed immediately."
    if score >= 50:
        return "Your mailbox is mostly usable, but suspicious messages require careful review."
    if score >= 25:
        return "Your mailbox is generally safe, but some messages deserve attention."
    if high_or_critical:
        return "Your mailbox has isolated high-risk findings despite a low aggregate score."
    return "Your mailbox appears safe based on the latest analyzed emails."


def analyze_latest_gmail(credential: Any = None, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Analyze the latest Gmail messages and return a deterministic mailbox security diagnosis."""
    fetched = gmail_fetch_tool(credential=credential, tool_context=tool_context)
    if "emails" not in fetched:
        return fetched

    emails = fetched.get("emails", [])
    reports = [_analyze_email_security(email, tool_context) for email in emails]
    emails_analyzed = len(reports)
    safe_emails = sum(1 for report in reports if report["risk_level"] == "SAFE")
    suspicious_emails = sum(1 for report in reports if report["risk_level"] in {"LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL"})
    high_risk_reports = [report for report in reports if report["risk_level"] in {"HIGH RISK", "CRITICAL"}]
    score = min(100, max((report["risk_score"] for report in reports), default=0) + min(10, max(0, suspicious_emails - 1) * 2))
    summary = {
        "emails_analyzed": emails_analyzed,
        "safe_emails": safe_emails,
        "suspicious_emails": suspicious_emails,
        "high_risk_emails": len(high_risk_reports),
    }
    findings = _mailbox_findings(reports)
    recommendations = _mailbox_recommendations(reports)
    result = {
        "overall_status": _overall_status(score),
        "security_score": score,
        "risk_level": _risk_level(score),
        "summary": summary,
        "findings": findings,
        "high_risk_emails": high_risk_reports,
        "recommendations": recommendations,
        "mailbox_diagnosis": _mailbox_diagnosis(summary, score, reports),
        "system_report": _system_report(summary, score, reports),
        "email_reports": reports,
    }
    state = _state_from_context(tool_context)
    state["last_mailbox_diagnosis"] = {
        "overall_status": result["overall_status"],
        "security_score": result["security_score"],
        "risk_level": result["risk_level"],
        "summary": summary,
    }
    _append_state_list(state, "mailbox_diagnosis_history", state["last_mailbox_diagnosis"])
    return result


def _gmail_fetch_adk_tool() -> AuthenticatedFunctionTool:
    return AuthenticatedFunctionTool(
        func=gmail_fetch_tool,
        auth_config=_gmail_auth_config(),
        response_for_auth_required={
            "status": "auth_required",
            "message": "Connect Gmail with Google OAuth to fetch messages.",
            "scope": GMAIL_READONLY_SCOPE,
        },
    )


def _gmail_analysis_adk_tool() -> AuthenticatedFunctionTool:
    return AuthenticatedFunctionTool(
        func=analyze_latest_gmail,
        auth_config=_gmail_auth_config(),
        response_for_auth_required={
            "status": "auth_required",
            "message": "Connect Gmail with Google OAuth to run mailbox security diagnosis.",
            "scope": GMAIL_READONLY_SCOPE,
        },
    )


def _recommended_actions(verdict: str) -> list[str]:
    if verdict == "high_risk":
        return [
            "Do not open the link.",
            "Quarantine or report it.",
            "Ask the agent to explain the model evidence before overriding.",
        ]
    if verdict == "needs_review":
        return [
            "Request human review.",
            "Run a second analysis with more context.",
        ]
    return [
        "Proceed with normal caution.",
        "Keep the analysis in session memory for comparison.",
    ]


def _build_model(prefer_fallback: bool = False) -> Any:
    if prefer_fallback:
        if LiteLlm is None:
            raise RuntimeError("LiteLlm support is unavailable; install google-adk[extensions].")
        return LiteLlm(model=os.getenv("GROQ_MODEL", "groq/llama-3.1-70b-versatile"))
    return os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def create_agent(prefer_fallback: bool = False) -> Agent:
    return Agent(
        name="tri_model_react_agent",
        model=_build_model(prefer_fallback=prefer_fallback),
        description="A ReAct-style agent that uses local classifier, spam, and embedding tools.",
        instruction=(
            "You are a ReAct-style assistant. Use tools rather than guessing. "
            "Use analyze_threat for hackathon demos, security triage, and any end-to-end risk verdict. "
            "Use route_request when the user asks for an end-to-end decision. "
            "Start with classify_request when the user's intent is unclear or when you need a label. "
            "Use detect_url_spam for URLs or suspicious links. "
            "Use decide_url_threat for URL threat-intelligence decisions that need grounded evidence, RDAP, Safe Browsing, and citations. "
            "Use analyze_embedding for semantic similarity, clustering, retrieval, or analysis. "
            "Use gmail_fetch_tool only after explicit Gmail OAuth consent when the user asks to fetch Gmail messages. "
            "Use analyze_latest_gmail for mailbox security diagnosis; report only deterministic scores, statuses, findings, and evidence from the tool output. "
            "Use remember_user_fact and search_session_memory when the user asks you to remember or compare prior analyses. "
            "Never claim a URL is safe only because a source has no match; summarize the structured evidence and unknowns."
        ),
        tools=[
            LoadMemoryTool(),
            _gmail_fetch_adk_tool(),
            _gmail_analysis_adk_tool(),
            decide_url_threat,
            analyze_threat,
            remember_user_fact,
            search_session_memory,
            route_request,
            classify_request,
            detect_url_spam,
            analyze_embedding,
        ],
    )


root_agent = create_agent()
try:
    fallback_agent = create_agent(prefer_fallback=True)
except RuntimeError:
    fallback_agent = None


def create_runner() -> Runner:
    """Create an ADK runner with in-memory session and memory services for local demos."""
    return Runner(
        agent=root_agent,
        app_name="tri_model_agent",
        session_service=SESSION_SERVICE,
        memory_service=MEMORY_SERVICE,
        auto_create_session=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tri-model router locally.")
    parser.add_argument("--threat", action="store_true", help="Run the hackathon threat verdict pipeline.")
    parser.add_argument("--decision", action="store_true", help="Run the grounded URL threat-intelligence decision tool.")
    parser.add_argument("text", nargs="*", help="Text or URL to route through the local models.")
    args = parser.parse_args()

    if not args.text:
        print("root_agent is ready. Import main.root_agent or tri_model_agent.agent.root_agent.")
        return

    text = " ".join(args.text)
    if args.decision:
        result = decide_url_threat(text)
    elif args.threat:
        result = analyze_threat(text)
    else:
        result = route_request(text)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
