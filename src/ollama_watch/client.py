"""Minimal read-only client for the Ollama HTTP API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime

DEFAULT_HOST = "127.0.0.1:11434"
#: Ollama reports an indefinite keep-alive as a sentinel far-future date.
FOREVER_HORIZON_S = 365 * 24 * 3600


@dataclass
class ModelInfo:
    """A loaded model, as reported by /api/ps."""

    name: str
    size_bytes: int = 0
    size_vram_bytes: int = 0
    context_length: int | None = None
    expires_in_s: float | None = None
    forever: bool = False

    @property
    def gpu_fraction(self) -> float | None:
        if not self.size_bytes:
            return None
        return self.size_vram_bytes / self.size_bytes


def fetch_loaded_model(host: str = DEFAULT_HOST, timeout: float = 2.0) -> ModelInfo | None:
    """Return the first loaded model, or None if the server is down or idle."""
    try:
        with urllib.request.urlopen(f"http://{host}/api/ps", timeout=timeout) as response:
            models = json.load(response).get("models") or []
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError, TimeoutError):
        return None
    if not models:
        return None

    raw = models[0]
    info = ModelInfo(
        name=raw.get("name") or raw.get("model") or "?",
        size_bytes=raw.get("size") or 0,
        size_vram_bytes=raw.get("size_vram") or 0,
        context_length=raw.get("context_length"),
    )
    expires = raw.get("expires_at") or ""
    try:
        remaining = datetime.fromisoformat(expires).timestamp() - datetime.now().timestamp()
    except ValueError:
        return info
    if remaining > FOREVER_HORIZON_S or expires.startswith(("0001-01-01", "9999")):
        info.forever = True
    else:
        info.expires_in_s = max(0.0, remaining)
    return info
