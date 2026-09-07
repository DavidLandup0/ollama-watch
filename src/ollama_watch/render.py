from __future__ import annotations

import os
import shutil
from datetime import datetime

from .client import ModelInfo
from .state import Phase, Receipt, RequestState

#: The code says "prefill", as the runner does; the display says what that is.
PHASE_LABELS = {
    Phase.IDLE: "Idle",
    Phase.PREFILL: "Processing input",
    Phase.DECODE: "Generating",
}

BAR_MIN, BAR_MAX = 10, 28
BAR_FILLED, BAR_EMPTY = "█", "░"
#: Divides a receipt into groups: status | input | output | memory.
SEPARATOR = "|"


class Style:
    """ANSI styling that collapses to plain text when unsupported."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def supports_color(stream) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def human_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 10_000 else str(count)


def bar(fraction: float, width: int) -> str:
    fraction = min(1.0, max(0.0, fraction))
    filled = int(fraction * width)
    return BAR_FILLED * filled + BAR_EMPTY * (width - filled)


def segmented_bar(parts: list[tuple[str, float]], width: int) -> str:
    """A bar divided proportionally between `(glyph, weight)` parts.

    Widths come from cumulative rounded edges, so the bar always fills exactly
    `width`; a part too small to earn a cell then borrows one from the largest
    rather than vanishing.
    """
    present = [(glyph, weight) for glyph, weight in parts if weight > 0]
    total = sum(weight for _, weight in present)
    if not total or width <= 0:
        return " " * max(0, width)

    edges = []
    running = 0.0
    for _, weight in present:
        running += weight
        edges.append(round(running / total * width))
    counts = [end - start for start, end in zip([0] + edges, edges)]

    for index, count in enumerate(counts):
        if count == 0 and max(counts) > 1:  # no room to share below that
            counts[counts.index(max(counts))] -= 1
            counts[index] = 1

    return "".join(glyph * count for (glyph, _), count in zip(present, counts))


def median(values) -> float | None:
    """The middle value, or None if there are none."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


def session_bar(session, width: int) -> str:
    """The session's time-accounting bar, with its legend."""
    segments = session.segments()
    legend = "  ".join(
        f"{glyph} {label} {human_duration(seconds)}"
        for glyph, label, seconds in segments
        if seconds
    )
    bar = segmented_bar([(glyph, seconds) for glyph, _, seconds in segments], width)
    return f"{bar}  {legend}"


def format_model(
    model: ModelInfo | None, style: Style, *, prompt_tokens: int = 0
) -> str:
    """Describe the loaded model.

    `context_length` is what /api/ps *declares*; the MLX engine grows its KV
    cache past it rather than truncating, so an over-long prompt is shown as
    `ctx<prompt` instead of implying a limit that is not enforced.
    """
    if model is None:
        return style.dim("[no model loaded]")
    parts = [style.cyan(model.name)]
    if model.size_bytes:
        parts.append(f"{model.size_bytes / 1e9:.0f}GB")
    gpu = model.gpu_fraction
    if gpu is not None:
        label = f"{gpu * 100:.0f}%GPU"
        parts.append(style.green(label) if gpu >= 0.99 else style.yellow(label))
    if model.context_length:
        if prompt_tokens > model.context_length:
            parts.append(style.yellow(f"ctx{model.context_length}<{human_tokens(prompt_tokens)}"))
        else:
            parts.append(f"ctx{model.context_length}")
    if model.forever:
        parts.append(style.green("∞"))
    elif model.expires_in_s is not None:
        parts.append(style.yellow(f"{human_duration(model.expires_in_s)} left"))
    return "[" + " ".join(parts) + "]"


def bar_width(columns: int | None = None) -> int:
    columns = columns or shutil.get_terminal_size((100, 24)).columns
    return max(BAR_MIN, min(BAR_MAX, columns - 74))


def format_status(
    state: RequestState,
    model: ModelInfo | None,
    style: Style,
    *,
    now: float | None = None,
    last_decode_rate: float | None = None,
    last_decode_exact: bool = False,
    columns: int | None = None,
) -> str:
    """One line describing what the server is doing right now."""
    head = format_model(model, style, prompt_tokens=state.prompt_tokens)

    if state.phase is Phase.IDLE:
        return f"{head} {style.dim(PHASE_LABELS[Phase.IDLE])} {style.dim('waiting for request')}"

    if state.phase is Phase.PREFILL:
        parts = [
            head,
            style.magenta(PHASE_LABELS[Phase.PREFILL]),
            style.bold(bar(state.fraction, bar_width(columns))),
            f"{state.fraction * 100:5.1f}%",
            style.dim(
                f"{human_tokens(state.done_tokens)}/{human_tokens(state.prompt_tokens)}"
                + ("" if state.cached_known else " new")
            ),
        ]
        if state.cached_tokens:
            parts.append(style.green(f"cached {human_tokens(state.cached_tokens)}"))
        rate = state.prefill_rate
        if rate:
            parts.append(f"{rate:.0f} tok/s")
            eta = state.eta_s
            if eta is not None:
                parts.append(style.yellow(f"eta {human_duration(eta)}"))
        return " ".join(parts)

    parts = [
        head,
        style.green(PHASE_LABELS[Phase.DECODE]),
        human_duration(state.decode_elapsed_s(now)),
    ]
    if state.generated_tokens:
        parts.append(style.dim(f"out {human_tokens(state.generated_tokens)}"))
    if last_decode_rate:
        marker = "" if last_decode_exact else "~"
        parts.append(style.dim(f"last {marker}{last_decode_rate:.0f} tok/s"))
    if state.prompt_tokens:  # unknown when we joined the request mid-decode
        parts.append(style.dim(f"input {human_tokens(state.prompt_tokens)}"))
    return " ".join(parts)


def format_receipt(receipt: Receipt, style: Style) -> str:
    """One line summarising a finished request."""
    if receipt.ok:
        mark = style.green("✔")
    elif receipt.status:
        mark = style.red("✖")
    else:
        mark = style.yellow("●")

    parts = [
        mark,
        datetime.fromtimestamp(receipt.ts).strftime("%H:%M:%S"),
        style.bold(receipt.status) if receipt.status else receipt.outcome,
    ]
    if receipt.raw_duration:
        parts.append(receipt.raw_duration)
    parts.append(style.dim(SEPARATOR))
    parts.append(f"input {human_tokens(receipt.prompt_tokens)}")
    if receipt.cached_tokens:
        parts.append(style.green(f"cached {human_tokens(receipt.cached_tokens)}"))

    if receipt.fully_cached:
        parts.append(style.green("fully cached"))
    else:
        parts.append(f"processed {human_tokens(receipt.prefilled_tokens)}")
        suffix = "" if receipt.prefill_trusted else "?"
        parts.append(f"@ {receipt.prefill_rate:.0f} tok/s{suffix}")
        parts.append(style.dim(f"({human_duration(receipt.prefill_s)})"))

    if receipt.generated_tokens is not None:
        marker = "" if receipt.generated_exact else "~"
        parts.append(style.dim(SEPARATOR))
        parts.append(f"out {marker}{human_tokens(receipt.generated_tokens)}")
        if receipt.decode_rate:
            parts.append(f"@ {marker}{receipt.decode_rate:.0f} tok/s")
        if receipt.decode_s:
            parts.append(style.dim(f"({human_duration(receipt.decode_s)})"))
        if receipt.acceptance is not None:
            parts.append(style.dim(f"mtp_accept_rate {receipt.acceptance:.2f}"))

    if receipt.peak_memory:
        parts.append(style.dim(SEPARATOR))
        parts.append(style.dim(f"peak {receipt.peak_memory}"))
    if receipt.outcome != "completed" and receipt.status is None:
        parts.append(style.red(f"[{receipt.outcome}]"))
    return " ".join(parts)
