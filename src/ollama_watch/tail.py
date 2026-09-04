"""Following the runner log, including restarts and mid-flight starts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

DEFAULT_LOG = os.path.expanduser("~/.ollama/logs/server.log")
#: How far back to look for the cache verdict of a request already in flight.
SEED_BYTES = 1 << 20
POLL_INTERVAL_S = 0.25

_CACHE_MARKERS = ('msg="cache hit"', 'msg="cache miss"')


@dataclass(frozen=True)
class LogLine:
    text: str
    #: True for lines replayed to rebuild state; consumers should not report
    #: these as if they had just happened.
    seeded: bool = False
    #: True for the marker emitted when the log is replaced (server restart).
    rotated: bool = False


def _seed_lines(handle, window: int = SEED_BYTES) -> list[str]:
    """Lines from the most recent cache verdict to the end of the file.

    Starting to follow a log mid-request would otherwise miss the verdict that
    says how much of the prompt was already cached, leaving only the relative
    remaining count to report.
    """
    end = handle.tell()
    start = max(0, end - window)
    handle.seek(start)
    chunk = handle.read(end - start)
    handle.seek(end)
    if start:  # a partial first line is unparseable
        newline = chunk.find("\n")
        chunk = chunk[newline + 1 :] if newline >= 0 else ""
    lines = chunk.splitlines()
    anchor = None
    for index, line in enumerate(lines):
        if any(marker in line for marker in _CACHE_MARKERS):
            anchor = index
    return lines[anchor:] if anchor is not None else []


def follow_lines(
    path: str = DEFAULT_LOG,
    *,
    replay: bool = False,
    follow: bool = True,
    poll_interval: float = POLL_INTERVAL_S,
    seed: bool = True,
) -> Iterator[LogLine]:
    """Yield log lines, seeding state and surviving rotation.

    With `replay`, starts at the beginning of the file. Otherwise starts at the
    end, first replaying the current request's lines as `seeded`. When `follow`
    is false the iterator stops at end of file; otherwise it polls forever and
    reopens the file when the server rotates it.
    """
    while not os.path.exists(path):
        if not follow:
            return
        time.sleep(poll_interval)

    handle = open(path, "r", errors="replace")
    inode = os.fstat(handle.fileno()).st_ino
    try:
        if not replay:
            handle.seek(0, os.SEEK_END)
            if seed:
                for text in _seed_lines(handle):
                    yield LogLine(text, seeded=True)

        pending = ""
        while True:
            chunk = handle.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n"):
                    yield LogLine(pending.rstrip("\n"))
                    pending = ""
                continue

            if not follow:
                return
            yield LogLine("", rotated=False)  # idle beat, lets callers tick
            try:
                stat = os.stat(path)
                if stat.st_ino != inode or stat.st_size < handle.tell():
                    handle.close()
                    handle = open(path, "r", errors="replace")
                    inode = os.fstat(handle.fileno()).st_ino
                    yield LogLine("", rotated=True)
            except FileNotFoundError:
                pass
            time.sleep(poll_interval)
    finally:
        handle.close()
