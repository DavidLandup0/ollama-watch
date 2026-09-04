"""A btop-style full-screen view. Standard-library curses; no dependencies.

Drawing is driven straight off `watch()`, which yields an idle beat every poll
interval, so there is no second loop and no threads.
"""

from __future__ import annotations

import curses
import os
import re
from collections import deque
from datetime import datetime

from .render import human_duration, human_tokens
from .state import Phase, Receipt
from .watch import watch

SPARK = "▁▂▃▄▅▆▇█"
FULL, EMPTY = "█", "░"
HISTORY = 48
RE_SIZE = re.compile(r"([\d.]+)\s*([KMGT]i?)B", re.I)
_UNITS = {"k": 1e3, "ki": 1024, "m": 1e6, "mi": 1024**2, "g": 1e9, "gi": 1024**3}

CYAN, GREEN, YELLOW, RED, MAGENTA = range(1, 6)


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
        for receipt in update.receipts:
            self.receipts.append(receipt)
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
        self._memory(width)
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
            self.field("status", "idle, waiting for request", curses.A_DIM)
            self.field("", gauge(0.0, bar_width), curses.A_DIM)
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

    def _rates(self, update, width: int) -> None:
        spark_width = max(8, width - 40)
        live = update.state.prefill_rate
        current = f"{live:.0f} tok/s" if live else "--"
        self.field(
            "input",
            f"{current.rjust(9)}  {spark(list(self.input_rates), spark_width)}",
            colour(MAGENTA),
        )
        last = update.last_decode_rate
        self.field(
            "output",
            f"{(f'{last:.0f} tok/s' if last else '--').rjust(9)}  "
            f"{spark(list(self.output_rates), spark_width)}",
            colour(GREEN),
        )

    def _memory(self, width: int) -> None:
        if not self.peak_bytes:
            return
        bar_width = max(10, width - 44)
        text = f"{self.peak_bytes / 1024**3:.1f} GiB peak"
        if self.total_bytes:
            fraction = self.peak_bytes / self.total_bytes
            pair = RED if fraction > 0.9 else YELLOW if fraction > 0.75 else GREEN
            self.field(
                "memory",
                f"{gauge(fraction, bar_width)} {fraction * 100:.0f}% of "
                f"{self.total_bytes / 1024**3:.0f} GiB  {text}",
                colour(pair),
            )
        else:
            self.field("memory", text)

    def _recent(self, width: int) -> None:
        height, _ = self.screen.getmaxyx()
        room = max(0, height - self.row - 2)
        if not room:
            return
        self.line(" recent", curses.A_BOLD)
        room -= 1
        for receipt in list(self.receipts)[-room:]:
            mark, pair = ("+", GREEN) if receipt.ok else ("!", RED) if receipt.status else ("-", YELLOW)
            fields = [
                mark,
                datetime.fromtimestamp(receipt.ts).strftime("%H:%M:%S"),
                (receipt.status or receipt.outcome).ljust(4),
                f"in {human_tokens(receipt.prompt_tokens).rjust(6)}",
                f"@ {receipt.prefill_rate:3.0f} tok/s",
            ]
            if receipt.generated_tokens is not None:
                fields.append(f"out {human_tokens(receipt.generated_tokens).rjust(6)}")
            if receipt.decode_rate:
                fields.append(f"@ {receipt.decode_rate:3.0f} tok/s")
            self.line(" " + "  ".join(fields), colour(pair))

    def _footer(self, update, width: int) -> None:
        height, _ = self.screen.getmaxyx()
        self.row = max(self.row, height - 1)
        total = sum(r.prompt_tokens for r in self.receipts)
        cached = sum(r.cached_tokens for r in self.receipts)
        failed = sum(1 for r in self.receipts if r.status and not r.ok)
        summary = [f"{len(self.receipts)} req"]
        if total:
            summary.append(f"cache {100 * cached / total:.0f}%")
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
