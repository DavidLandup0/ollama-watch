"""Live input-processing and generation progress for a local Ollama server.

Ollama's HTTP API sends nothing between a request and its first token, so no
client can show how far along a long prompt is. The runner logs that progress;
this package parses it into events, folds them into request state, and renders
it -- or hands it to you.

    from ollama_watch import watch, Phase

    for update in watch():
        if update.state.phase is Phase.PREFILL:
            print(update.state.fraction, update.state.eta_s)
        for receipt in update.receipts:
            print(receipt.prompt_tokens, receipt.decode_rate)
"""

from .client import DEFAULT_HOST, ModelInfo, fetch_loaded_model
from .dashboard import Dashboard, gauge, spark
from .events import (
    PREFILL_BATCH,
    CacheVerdict,
    Event,
    PeakMemory,
    Progress,
    RequestEnd,
    RunnerReady,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)
from .parse import dur_seconds, parse_kv, parse_line
from .render import PHASE_LABELS, format_receipt, format_status
from .state import Phase, Receipt, RequestState, Tracker
from .tail import DEFAULT_LOG, LogLine, follow_lines
from .watch import Update, watch

__version__ = "0.1.0"

__all__ = [
    "CacheVerdict",
    "Dashboard",
    "DEFAULT_HOST",
    "DEFAULT_LOG",
    "Event",
    "LogLine",
    "ModelInfo",
    "PHASE_LABELS",
    "PREFILL_BATCH",
    "PeakMemory",
    "Phase",
    "Progress",
    "Receipt",
    "RequestEnd",
    "RequestState",
    "RunnerReady",
    "SlotTiming",
    "SpeculativeStats",
    "Terminated",
    "Tracker",
    "Update",
    "__version__",
    "dur_seconds",
    "fetch_loaded_model",
    "follow_lines",
    "gauge",
    "format_receipt",
    "format_status",
    "parse_kv",
    "parse_line",
    "spark",
    "watch",
]
