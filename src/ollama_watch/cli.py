"""Command line front end: a live status line plus per-request receipts."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from .client import DEFAULT_HOST
from .render import Style, format_receipt, format_status, human_tokens, supports_color
from .state import Phase, Receipt
from .tail import DEFAULT_LOG
from .watch import PS_INTERVAL_S, watch


class Display:
    """A single self-updating status line beneath scrolling receipts.

    On a TTY the status line is redrawn in place. Piped, it prints a plain line
    whenever progress actually moves, so the output stays useful in a file.
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

    prefill = sorted(r.prefill_rate for r in receipts if r.prefill_trusted and r.prefill_rate > 0)
    if prefill:
        lines.append(
            f"  input rate     median {prefill[len(prefill) // 2]:.0f} tok/s   "
            f"min {prefill[0]:.0f}   max {prefill[-1]:.0f}"
        )
    decode = sorted(r.decode_rate for r in receipts if r.decode_rate)
    if decode:
        lines.append(
            f"  output rate    median {decode[len(decode) // 2]:.0f} tok/s   "
            f"min {decode[0]:.0f}   max {decode[-1]:.0f}"
        )
    total = sum(r.prompt_tokens for r in receipts)
    cached = sum(r.cached_tokens for r in receipts)
    if total:
        lines.append(
            f"  input tokens   {human_tokens(total)} total, "
            f"{human_tokens(cached)} from cache ({100 * cached / total:.0f}%)"
        )
    generated = sum(r.generated_tokens or 0 for r in receipts)
    if generated:
        lines.append(f"  output tokens  {human_tokens(generated)} total")
    failed = [r for r in receipts if r.status and not r.ok]
    if failed:
        codes = ", ".join(sorted({r.status for r in failed if r.status}))
        lines.append(style.red(f"  failed         {len(failed)} request(s): {codes}"))
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
