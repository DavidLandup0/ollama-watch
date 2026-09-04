"""A btop-style full-screen view. Standard-library curses; no dependencies.

Drawing is driven straight off `watch()`, which yields an idle beat every poll
interval, so there is no second loop and no threads.
"""

from __future__ import annotations

import curses
import os
import re
import time
from collections import deque
from datetime import datetime

from .render import human_duration, human_tokens, segmented_bar
from .session import Session
from .state import Phase, Receipt
from .watch import watch

SPARK = "▁▂▃▄▅▆▇█"
FULL, EMPTY = "█", "░"
HISTORY = 48
RE_SIZE = re.compile(r"([\d.]+)\s*([KMGT]i?)B", re.I)
_UNITS = {"k": 1e3, "ki": 1024, "m": 1e6, "mi": 1024**2, "g": 1e9, "gi": 1024**3}

CYAN, GREEN, YELLOW, RED, MAGENTA = range(1, 6)

#: The request table. Terminal themes cannot be relied on to distinguish
#: colours -- a pink monochrome theme renders red and green alike -- so status
#: is carried by a glyph and by fixed columns, never by hue alone.
COLUMNS = (
    ("time", 8, ">"),
    ("code", 8, ">"),
    ("input", 7, ">"),
    ("in/s", 5, ">"),
    ("cache", 6, ">"),
    ("out", 6, ">"),
    ("out/s", 6, ">"),
    ("queue", 6, ">"),
    ("ttft", 6, ">"),
    ("dur", 6, ">"),
    ("peak", 9, ">"),
)
MARKS = {"ok": "+", "fail": "!", "other": "-"}


def row(cells: list[str]) -> str:
    return "  ".join(f"{cell:{align}{width}}" for cell, (_, width, align) in zip(cells, COLUMNS))


def colour(pair: int) -> int:
    """A colour attribute, or plain text where colours are unavailable."""
    try:
        return curses.color_pair(pair)
    except curses.error:
        return 0


def spark(values: list[float], width: int) -> str:
    """A sparkline of the most recent `width` values, scaled to their own peak."""
    if not values:
        return ""
    recent = values[-width:]
    peak = max(recent)
    if peak <= 0:
        return SPARK[0] * len(recent)
    return "".join(SPARK[min(len(SPARK) - 1, int(v / peak * (len(SPARK) - 1)))] for v in recent)


def spark_live(history: list[float], live: float | None, width: int) -> str:
    """History, then a divider, then a cell for the request in flight.

    Both are scaled together so the live cell is comparable to the bars behind
    it. Without the divider a provisional value would be indistinguishable
    from a finished one.
    """
    if live is None:
        return spark(history, width)
    scaled = spark(history + [live], width + 1)
    if len(scaled) < 2:
        return scaled
    return f"{scaled[:-1]}\u2503{scaled[-1]}"


def gauge(fraction: float, width: int) -> str:
    fraction = min(1.0, max(0.0, fraction))
    filled = int(fraction * width)
    return FULL * filled + EMPTY * (width - filled)


def parse_size(text: str | None) -> float | None:
    """`"32.49 GiB"` -> bytes."""
    if not text:
        return None
    match = RE_SIZE.search(text)
    if not match:
        return None
    return float(match.group(1)) * _UNITS.get(match.group(2).lower(), 1)


def total_memory() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


class Dashboard:
    """Renders one frame per update. Holds only what the frame needs."""

    def __init__(self, screen) -> None:
        self.screen = screen
        self.input_rates: deque[float] = deque(maxlen=HISTORY)
        self.output_rates: deque[float] = deque(maxlen=HISTORY)
        self.receipts: deque[Receipt] = deque(maxlen=HISTORY)
        self.peak_bytes: float | None = None
        self.session = Session()
        self.total_bytes = total_memory()
        self.row = 0

    # ---- drawing primitives ----

    def line(self, text: str = "", attr: int = 0) -> None:
        height, width = self.screen.getmaxyx()
        if self.row >= height:
            return
        try:
            self.screen.addnstr(self.row, 0, text.ljust(width - 1), width - 1, attr)
        except curses.error:
            pass
        self.row += 1

    def field(self, label: str, value: str, attr: int = 0, *, pad: int = 8) -> None:
        self.line(f" {label.ljust(pad)}{value}", attr)

    # ---- frame ----

    def draw(self, update) -> None:
        self.screen.erase()
        self.row = 0
        _, width = self.screen.getmaxyx()
        self.session.observe(update.event.ts if update.event else time.time())
        for receipt in update.receipts:
            self.receipts.append(receipt)
            self.session.add(receipt)
            if receipt.prefill_trusted and receipt.prefill_rate > 0:
                self.input_rates.append(receipt.prefill_rate)
            if receipt.decode_rate:
                self.output_rates.append(receipt.decode_rate)
            self.peak_bytes = parse_size(receipt.peak_memory) or self.peak_bytes

        self._header(update, width)
        self.line()
        self._current(update, width)
        self.line()
        self._rates(update, width)
        self.line()
        self._memory(update, width)
        self.line()
        self._session(width)
        self.line()
        self._recent(width)
        self._footer(update, width)
        self.screen.refresh()

    def _header(self, update, width: int) -> None:
        model = update.model
        self.line(" ollama-watch", curses.A_BOLD)
        if model is None:
            self.field("model", "no model loaded", curses.A_DIM)
            return
        bits = [model.name]
        if model.size_bytes:
            bits.append(f"{model.size_bytes / 1e9:.0f}GB")
        gpu = model.gpu_fraction
        if gpu is not None:
            bits.append(f"{gpu * 100:.0f}% GPU")
        if model.context_length:
            declared = f"ctx{model.context_length}"
            prompt = update.state.prompt_tokens
            bits.append(
                f"{declared}<{human_tokens(prompt)}" if prompt > model.context_length else declared
            )
        bits.append("∞" if model.forever else human_duration(model.expires_in_s) + " left")
        self.field("model", "  ".join(bits), colour(CYAN))

    def _current(self, update, width: int) -> None:
        state = update.state
        bar_width = max(10, width - 44)
        if state.phase is Phase.IDLE:
            last = self.receipts[-1] if self.receipts else None
            detail = "waiting for request"
            if last is not None:
                ago = max(0.0, time.time() - last.ts)
                detail += f"   last {human_duration(ago)} ago"
            self.field("status", f"idle   {detail}", curses.A_DIM)
            return

        if state.phase is Phase.PREFILL:
            total = human_tokens(state.prompt_tokens) + ("" if state.cached_known else " new")
            self.field("status", "Processing input", colour(MAGENTA) | curses.A_BOLD)
            self.field(
                "",
                f"{gauge(state.fraction, bar_width)} {state.fraction * 100:5.1f}%  "
                f"{human_tokens(state.done_tokens)}/{total}",
            )
            detail = []
            if state.cached_tokens:
                detail.append(f"cached {human_tokens(state.cached_tokens)}")
            if state.eta_s is not None:
                detail.append(f"eta {human_duration(state.eta_s)}")
            if state.started_at:
                # no token has been emitted yet, so this is TTFT so far
                detail.append(f"ttft so far {human_duration(time.time() - state.started_at)}")
            self.field("", "  ".join(detail), curses.A_DIM)
            return

        self.field("status", "Generating", colour(GREEN) | curses.A_BOLD)
        self.field("", f"{gauge(1.0, bar_width)} {human_duration(state.decode_elapsed_s())}")
        reference = (
            f"last {'' if update.last_decode_exact else '~'}{update.last_decode_rate:.0f} tok/s"
            if update.last_decode_rate
            else "no rate reported yet"
        )
        self.field("", f"input {human_tokens(state.prompt_tokens)}  {reference}", curses.A_DIM)

    def _rate_row(
        self,
        label: str,
        current: float | None,
        tag: str,
        history: list[float],
        width: int,
        pair: int,
        *,
        live: float | None = None,
    ) -> None:
        spark_width = max(8, min(len(history) or 1, width - 52))
        value = f"{current:.0f} tok/s" if current else "--"
        text = f"{value.rjust(9)} {tag.ljust(5)} {spark_live(history, live, spark_width)}"
        if len(history) >= 2:
            text += f"  {min(history):.0f}-{max(history):.0f} over {len(history)}"
        self.field(label, text, colour(pair))

    def _rates(self, update, width: int) -> None:
        """Sparklines are one bar per finished request.

        The numbers beside them mean different things by phase, and say which:
        the input rate is live only while input is being processed, and the
        generation rate is never live -- nothing is logged per token, so it is
        always the last request that reported one.
        """
        state = update.state
        last = self.receipts[-1] if self.receipts else None
        if state.phase is Phase.PREFILL:
            rate, tag, live = state.prefill_rate, "now", state.prefill_rate
        elif state.phase is Phase.DECODE:
            rate, tag, live = state.prefill_rate, "req", state.prefill_rate
        else:
            rate, tag, live = (last.prefill_rate if last else None), "last", None
        self._rate_row("input", rate, tag, list(self.input_rates), width, MAGENTA, live=live)
        self._rate_row(
            "output", update.last_decode_rate, "last", list(self.output_rates), width, GREEN
        )

    def _memory(self, update, width: int) -> None:
        """Resident size against physical RAM, with peak marked in the bar.

        The bar tracks what the model holds *now*; the peak is a high-water
        mark from a past request and would otherwise sit at full width forever.
        """
        resident = float(update.model.size_vram_bytes or update.model.size_bytes) if update.model else 0.0
        if not resident and not self.peak_bytes:
            return
        bar_width = max(10, min(40, width - 62))
        if not self.total_bytes:
            self.field("memory", f"{resident / 1024**3:.1f} GiB resident")
            return

        fraction = resident / self.total_bytes
        peak_fraction = (self.peak_bytes or 0) / self.total_bytes
        bar = list(gauge(fraction, bar_width))
        marker = int(peak_fraction * bar_width)
        if 0 <= marker < bar_width:
            bar[marker] = "┊"
        pair = RED if peak_fraction > 1.0 else YELLOW if peak_fraction > 0.85 else GREEN
        detail = f"{resident / 1024**3:.1f} of {self.total_bytes / 1024**3:.0f} GiB"
        if self.peak_bytes:
            detail += f"   peak {self.peak_bytes / 1024**3:.1f} ({peak_fraction * 100:.0f}%)"
        self.field("memory", f"{''.join(bar)}  {detail}", colour(pair))

    @staticmethod
    def _cells(receipt: Receipt) -> list[str]:
        code = receipt.status or receipt.outcome.split(":")[0][:8]
        approx = "" if receipt.generated_exact else "~"
        return [
            datetime.fromtimestamp(receipt.ts).strftime("%H:%M:%S"),
            code,
            human_tokens(receipt.prompt_tokens),
            f"{receipt.prefill_rate:.0f}" if receipt.prefill_rate else "",
            f"{receipt.cache_fraction * 100:.0f}%" if receipt.cached_tokens else "",
            f"{approx}{human_tokens(receipt.generated_tokens)}" if receipt.generated_tokens else "",
            f"{approx}{receipt.decode_rate:.0f}" if receipt.decode_rate else "",
            human_duration(receipt.queued_s) if receipt.queued_s else "",
            human_duration(receipt.ttft_s) if receipt.ttft_s else "",
            human_duration(receipt.duration_s) if receipt.duration_s else "",
            f"{parse_size(receipt.peak_memory) / 1024**3:.1f} GiB" if receipt.peak_memory else "",
        ]

    def _session(self, width: int) -> None:
        """Where the wall clock went: model working versus waiting for you."""
        session = self.session
        if not session.span_s:
            return
        working = session.busy_s
        self.field(
            "session",
            f"{human_duration(session.span_s)}   "
            f"working {human_duration(working)} ({session.fraction(working) * 100:.0f}%)   "
            f"idle {human_duration(session.idle_s)} ({session.fraction(session.idle_s) * 100:.0f}%)",
            curses.A_BOLD,
        )
        segments = session.segments()
        bar_width = max(10, min(40, width - 62))
        legend = "  ".join(
            f"{glyph} {label} {human_duration(seconds)}" for glyph, label, seconds in segments if seconds
        )
        self.field("", f"{segmented_bar([(g, s) for g, _, s in segments], bar_width)}  {legend}")

    def _recent(self, width: int) -> None:
        height, _ = self.screen.getmaxyx()
        room = max(0, height - self.row - 2)
        if room < 3:
            return
        self.line("   " + row([header for header, _, _ in COLUMNS]), curses.A_DIM)
        room -= 1
        for receipt in list(self.receipts)[-room:]:
            key = "ok" if receipt.ok else "fail" if receipt.status else "other"
            pair = {"ok": GREEN, "fail": RED, "other": YELLOW}[key]
            self.line(f" {MARKS[key]} " + row(self._cells(receipt)), colour(pair))

    def _footer(self, update, width: int) -> None:
        height, _ = self.screen.getmaxyx()
        self.row = max(self.row, height - 1)
        total = sum(r.prompt_tokens for r in self.receipts)
        cached = sum(r.cached_tokens for r in self.receipts)
        failed = sum(1 for r in self.receipts if r.status and not r.ok)
        summary = [f"{len(self.receipts)} req"]
        if total:
            summary.append(f"cache {100 * cached / total:.0f}%")
        for label, history in (("in", self.input_rates), ("out", self.output_rates)):
            if history:
                ordered = sorted(history)
                summary.append(f"median {label} {ordered[len(ordered) // 2]:.0f} tok/s")
        ttfts = sorted(r.ttft_s for r in self.receipts if r.ttft_s)
        if ttfts:
            summary.append(f"median ttft {human_duration(ttfts[len(ttfts) // 2])}")
        if failed:
            summary.append(f"failed {failed}")
        summary.append("q quit")
        self.line(" " + "  ".join(summary), curses.A_DIM)


def _init_colours() -> None:
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return
    for pair, colour in (
        (CYAN, curses.COLOR_CYAN),
        (GREEN, curses.COLOR_GREEN),
        (YELLOW, curses.COLOR_YELLOW),
        (RED, curses.COLOR_RED),
        (MAGENTA, curses.COLOR_MAGENTA),
    ):
        try:
            curses.init_pair(pair, colour, -1)
        except curses.error:
            pass


def run(**kwargs) -> int:
    """Run the dashboard until the log ends or the user quits."""

    def loop(screen) -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        _init_colours()
        dashboard = Dashboard(screen)
        for update in watch(**kwargs):
            dashboard.draw(update)
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                return

    curses.wrapper(loop)
    return 0
