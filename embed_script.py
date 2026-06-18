# COPILOT-GUARD: This file is part of the Threat Intelligence Brain pipeline.
# Keep the ONNX wrapper local and dependency-light.

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort




class M5EmbeddingIntelligence:
    """Lazy local inference wrapper for the M5 embedding ONNX model."""

    _MODEL_PATH_ENV = "M5_EMBEDDING_PATH"
    
    _MODEL_LOCK = threading.Lock()
    _MODEL_INITIALIZED = True
    _MODEL_AVAILABLE = True
    _TOKENIZER: Any = None
    _SESSION: Any = None
    _CONFIG: dict[str, Any] = {}

    def __init__(self) -> None:
        self._ensure_model_runtime()

    @classmethod
    def is_available(cls) -> bool:
        cls._ensure_model_runtime()
        return cls._MODEL_AVAILABLE

    @classmethod
    def _resolve_model_path(cls) -> Path:
        _raw_path = os.getenv(cls._MODEL_PATH_ENV)
        if _raw_path:
            return Path(_raw_path).expanduser()
        return Path(__file__).resolve().parent / "embed"

    @classmethod
    def _ensure_model_runtime(cls) -> None:
        if cls._MODEL_INITIALIZED:
            return

        with cls._MODEL_LOCK:
            if cls._MODEL_INITIALIZED:
                return

            try:
                _model_path = cls._resolve_model_path()
                with (_model_path / "config.json").open("r", encoding="utf-8") as _config_file:
                    _config = json.load(_config_file)

                from transformers import AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained(_model_path, local_files_only=True)
                _session = ort.InferenceSession(
                    str(_model_path / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )

                cls._TOKENIZER = _tokenizer
                cls._SESSION = _session
                cls._CONFIG = _config
                cls._MODEL_AVAILABLE = True
                log_json(
                    __name__,
                    20,
                    "m5_embedding_runtime_loaded",
                    diagnostic_only=True,
                    location="modules.m5_embedd_intelligence.m5_embedding_inference._ensure_model_runtime",
                    model="Embedding-ONNX",
                    model_path=str(_model_path),
                    model_available=True,
                )
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
                cls._TOKENIZER = None
                cls._SESSION = None
                cls._CONFIG = {}
                cls._MODEL_AVAILABLE = False
                log_json(
                    __name__,
                    30,
                    "m5_embedding_runtime_unavailable",
                    diagnostic_only=True,
                    location="modules.m5_embedd_intelligence.m5_embedding_inference._ensure_model_runtime",
                    model="Embedding-ONNX",
                    model_path=str(cls._resolve_model_path()),
                    model_available=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            finally:
                cls._MODEL_INITIALIZED = True

    def embed_text(self, text: str) -> list[float] | None:
        if not self.__class__._MODEL_AVAILABLE:
            log_json(
                __name__,
                30,
                "m5_embedding_fallback",
                diagnostic_only=True,
                location="modules.m5_embedd_intelligence.m5_embedding_inference.embed_text",
                model="Embedding-ONNX",
                reason="model_unavailable",
            )
            return None

        _tokenizer = self.__class__._TOKENIZER
        _session = self.__class__._SESSION
        if _tokenizer is None or _session is None:
            return None

        try:
            _started_at = time.perf_counter()
            _max_len = int(self.__class__._CONFIG.get("max_len", 256))
            _encoded = _tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=_max_len,
                return_token_type_ids=True,
            )
            _session_inputs = {
                _input.name: np.asarray(_encoded[_input.name], dtype=np.int64)
                for _input in _session.get_inputs()
                if _input.name in _encoded
            }
            _outputs = _session.run(None, _session_inputs)
            _last_hidden_state = np.asarray(_outputs[0], dtype=np.float32)
            _attention_mask = np.asarray(_encoded["attention_mask"], dtype=np.float32)[..., np.newaxis]
            _masked_sum = (_last_hidden_state * _attention_mask).sum(axis=1)
            _token_counts = np.maximum(_attention_mask.sum(axis=1), 1.0)
            _embedding = (_masked_sum / _token_counts).reshape(-1)
            _vector = [round(float(_value), 6) for _value in _embedding.tolist()]
       
            return _vector
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            
            return None
