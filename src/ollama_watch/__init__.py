from .client import DEFAULT_HOST, ModelInfo, check_prerequisites, fetch_loaded_model, ping_server
from .dashboard import Dashboard, gauge, spark
from .events import (
    PREFILL_BATCH,
    CacheVerdict,
    DecodeTick,
    Event,
    PeakMemory,
    PrefillDone,
    Progress,
    RequestEnd,
    RunnerReady,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)
from .parse import (
    begins_request,
    dur_seconds,
    ends_request,
    line_ts,
    parse_kv,
    parse_line,
)
from .render import PHASE_LABELS, format_receipt, format_status, segmented_bar
from .session import Session
from .state import Phase, Receipt, RequestState, Tracker
from .tail import DEFAULT_LOG, LogLine, follow_lines
from .watch import Update, watch

__version__ = "0.1.0"

__all__ = [
    "CacheVerdict",
    "Dashboard",
    "DEFAULT_HOST",
    "DEFAULT_LOG",
    "DecodeTick",
    "Event",
    "LogLine",
    "ModelInfo",
    "PHASE_LABELS",
    "PREFILL_BATCH",
    "PeakMemory",
    "Phase",
    "PrefillDone",
    "Progress",
    "Receipt",
    "Session",
    "RequestEnd",
    "RequestState",
    "RunnerReady",
    "SlotTiming",
    "SpeculativeStats",
    "Terminated",
    "Tracker",
    "Update",
    "__version__",
    "begins_request",
    "dur_seconds",
    "check_prerequisites",
    "ends_request",
    "fetch_loaded_model",
    "follow_lines",
    "gauge",
    "format_receipt",
    "format_status",
    "line_ts",
    "parse_kv",
    "parse_line",
    "ping_server",
    "segmented_bar",
    "spark",
    "watch",
]
