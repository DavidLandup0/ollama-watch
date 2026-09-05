from __future__ import annotations

import argparse
import signal
import sys
import time

from .client import DEFAULT_HOST, check_prerequisites
from .render import (
    Style,
    format_receipt,
    format_status,
    human_duration,
    human_tokens,
    median,
    session_bar,
    supports_color,
)
from .session import Session
from .state import Receipt
from .tail import DEFAULT_LOG
from .watch import PS_INTERVAL_S, watch


class Display:
    """A single self-updating status line beneath scrolling receipts.

    Redrawn in place on a TTY; piped, printed afresh whenever progress moves.
    """

    def __init__(self, style: Style, *, live: bool, stream=sys.stdout) -> None:
        self.style = style
        self.live = live
        self.stream = stream
        self._line_open = False
        self._last_key: tuple | None = None

    def clear(self) -> None:
        if self._line_open:
            self.stream.write("\r\033[2K")
            self._line_open = False

    def note(self, text: str) -> None:
        self.clear()
        self.stream.write(text + "\n")
        self.stream.flush()

    def status(self, text: str, key: tuple) -> None:
        if not self.live:
            if key != self._last_key:
                self._last_key = key
                self.note(text)
            return
        self.clear()
        self.stream.write("\r\033[2K" + text)
        self.stream.flush()
        self._line_open = True


def summarise(receipts: list[Receipt], style: Style) -> list[str]:
    """Aggregate statistics over a session's receipts."""
    if not receipts:
        return []
    lines = ["", style.bold(f"{len(receipts)} request(s)")]

    def row(label: str, text: str) -> str:
        """One aligned `label   value` line."""
        return f"  {label:<13}  {text}"

    def rate_row(label: str, rates: list[float]) -> None:
        if rates:
            lines.append(
                row(label, f"median {median(rates):.0f} tok/s   min {min(rates):.0f}   max {max(rates):.0f}")
            )

    rate_row("input rate", [r.prefill_rate for r in receipts if r.prefill_trusted and r.prefill_rate > 0])
    rate_row("output rate", [r.decode_rate for r in receipts if r.decode_rate])

    ttfts = [r.ttft_s for r in receipts if r.ttft_s]
    if ttfts:
        text = f"median {human_duration(median(ttfts))}   max {human_duration(max(ttfts))}"
        queued = [r.queued_s for r in receipts if r.queued_s]
        if queued:
            text += f"   queued median {human_duration(median(queued))}"
        lines.append(row("ttft", text))

    total = sum(r.prompt_tokens for r in receipts)
    cached = sum(r.cached_tokens for r in receipts)
    if total:
        lines.append(
            row("input tokens", f"{human_tokens(total)} total, "
                               f"{human_tokens(cached)} from cache ({100 * cached / total:.0f}%)")
        )
    generated = sum(r.generated_tokens or 0 for r in receipts)
    if generated:
        lines.append(row("output tokens", f"{human_tokens(generated)} total"))

    session = Session()
    for receipt in receipts:
        session.add(receipt)
    if session.span_s:
        lines.append(
            row("session", f"{human_duration(session.span_s)} span, "
                           f"{human_duration(session.busy_s)} working "
                           f"({session.fraction(session.busy_s) * 100:.0f}%), "
                           f"{human_duration(session.idle_s)} idle "
                           f"({session.fraction(session.idle_s) * 100:.0f}%)")
        )
        lines.append(row("", session_bar(session, 32)))

    failed = [r for r in receipts if r.status and not r.ok]
    if failed:
        codes = ", ".join(sorted({r.status for r in failed if r.status}))
        lines.append(style.red(row("failed", f"{len(failed)} request(s): {codes}")))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-watch",
        description="Live input-processing and generation progress for a local Ollama server.",
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help=f"runner log (default: {DEFAULT_LOG})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"ollama host (default: {DEFAULT_HOST})")
    parser.add_argument("--replay", action="store_true", help="parse the log from the start")
    parser.add_argument(
        "--no-follow", action="store_true", help="exit at end of log instead of following"
    )
    parser.add_argument(
        "--no-server", action="store_true", help="do not query /api/ps for model details"
    )
    parser.add_argument(
        "-d", "--dashboard", action="store_true", help="full-screen dashboard view"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    error = check_prerequisites(args.log, args.host, query_server=not args.no_server)
    if error is not None:
        print(error, file=sys.stderr)
        return 1

    if args.dashboard:
        from .dashboard import run

        return run(
            log=args.log,
            host=args.host,
            replay=args.replay,
            follow=not args.no_follow,
            query_server=not args.no_server,
        )

    style = Style(supports_color(sys.stdout))
    display = Display(style, live=sys.stdout.isatty() and not args.no_follow)
    collected: list[Receipt] = []

    def finish(*_) -> None:
        display.clear()
        for line in summarise(collected, style):
            print(line)
        sys.exit(0)

    signal.signal(signal.SIGINT, finish)

    try:
        for update in watch(
            log=args.log,
            host=args.host,
            replay=args.replay,
            follow=not args.no_follow,
            ps_interval=PS_INTERVAL_S,
            query_server=not args.no_server,
        ):
            if update.rotated:
                display.note(style.dim("-- log rotated, reopening"))
                continue
            for receipt in update.receipts:
                collected.append(receipt)
                display.note(format_receipt(receipt, style))
            if update.seeded:
                continue

            state = update.state
            display.status(
                format_status(
                    state,
                    update.model,
                    style,
                    now=time.time(),
                    last_decode_rate=update.last_decode_rate,
                    last_decode_exact=update.last_decode_exact,
                ),
                key=(state.phase, state.done_tokens),
            )
    finally:
        display.clear()

    for line in summarise(collected, style):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
