from __future__ import annotations

from ollama_watch import Phase, Tracker, follow_lines, parse_line, watch

CACHE_HIT = (
    'time=2026-09-04T14:08:05.604+09:00 level=INFO source=prefix_cache.go:125 '
    'msg="cache hit" total=49091 matched=34816 cached=34816 left=14275'
)
PROGRESS_1 = (
    'time=2026-09-04T14:08:35.000+09:00 level=INFO source=pipeline.go:223 '
    'msg="Prompt processing progress" processed=2048 total=14275'
)
PROGRESS_2 = (
    'time=2026-09-04T14:09:05.000+09:00 level=INFO source=pipeline.go:223 '
    'msg="Prompt processing progress" processed=8192 total=14275'
)


def write_log(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def test_seeding_recovers_the_cached_prefix(tmp_path):
    """Starting to follow mid-request must not lose the cache verdict."""
    log = write_log(tmp_path / "server.log", ["unrelated noise", CACHE_HIT, PROGRESS_1, PROGRESS_2])
    seeded = [line for line in follow_lines(str(log), follow=False)]
    assert [line.seeded for line in seeded] == [True, True, True]

    tracker = Tracker()
    for line in seeded:
        event = parse_line(line.text)
        if event:
            tracker.feed(event)
    assert tracker.state.prompt_tokens == 49091
    assert tracker.state.done_tokens == 34816 + 8192


def test_replay_reads_from_the_start(tmp_path):
    log = write_log(tmp_path / "server.log", [CACHE_HIT, PROGRESS_1])
    lines = [line for line in follow_lines(str(log), replay=True, follow=False)]
    assert [line.seeded for line in lines] == [False, False]
    assert len(lines) == 2


def test_no_seed_anchor_yields_nothing(tmp_path):
    log = write_log(tmp_path / "server.log", ["nothing", "of", "interest"])
    assert [line for line in follow_lines(str(log), follow=False)] == []


def test_missing_log_is_not_fatal(tmp_path):
    assert list(follow_lines(str(tmp_path / "absent.log"), follow=False)) == []


def test_watch_yields_state_and_receipts(tmp_path):
    log = write_log(
        tmp_path / "server.log",
        [
            CACHE_HIT,
            PROGRESS_1,
            'time=2026-09-04T14:09:20.000+09:00 level=INFO source=pipeline.go:223 '
            'msg="Prompt processing progress" processed=14274 total=14275',
            '[GIN] 2026/09/04 - 14:09:30 | 200 |         85s |       127.0.0.1 | POST     "/v1/chat/completions"',
        ],
    )
    updates = list(
        watch(log=str(log), replay=True, follow=False, query_server=False)
    )
    assert updates, "watch produced no updates"
    phases = [u.state.phase for u in updates]
    assert Phase.PREFILL in phases
    receipts = [r for u in updates for r in u.receipts]
    assert len(receipts) == 1
    assert receipts[0].status == "200"
    assert receipts[0].cached_tokens == 34816
