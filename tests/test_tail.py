from __future__ import annotations

import pytest

from ollama_watch import Phase, Tracker, follow_lines, parse_line, watch
from ollama_watch.parse import parse_gin_ts

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


def test_seed_anchor_includes_llamacpp_new_prompt(tmp_path):
    """Joining a llama.cpp request mid-flight must replay from `new prompt`."""
    log = write_log(
        tmp_path / "server.log",
        [
            "srv  update_slots: all slots are idle",
            "slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 2050",
            "slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   1024, progress = 0.50, t =   7.78 s / 131.61 tokens per second",
        ],
    )
    seeded = [line for line in follow_lines(str(log), follow=False)]
    assert [line.seeded for line in seeded] == [True, True]

    tracker = Tracker()
    for line in seeded:
        event = parse_line(line.text, now=100.0)
        if event:
            tracker.feed(event)
    assert tracker.state.prompt_tokens == 2050
    assert tracker.state.phase is Phase.PREFILL


def _live_watch(path, **kwargs):
    """A following watcher on an existing-but-empty log."""
    path.write_text("")
    return watch(log=str(path), replay=False, follow=True, **kwargs)


def _append(path, *lines):
    with open(path, "a") as handle:
        handle.write("".join(line + "\n" for line in lines))


REQUEST_LINES = [
    "slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 2050",
    "slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   1024, progress = 0.50, t =   7.78 s / 131.61 tokens per second",
    "slot print_timing: id  0 | task 7 | n_gen =    100, tg =   4.80 t/s, tg_3s =   4.85 t/s",
]


def test_watch_terminates_request_when_server_vanishes(tmp_path):
    """An active request with no loaded model cannot complete; report it."""
    path = tmp_path / "server.log"
    gen = _live_watch(
        path,
        host="127.0.0.1:1",  # nothing listens: every poll misses
        poll_interval=0.01,
        ps_interval=0.03,
        query_server=True,
    )
    next(gen)  # idle beat on the empty log
    _append(path, *REQUEST_LINES)
    receipts = []
    for _ in range(500):
        try:
            update = next(gen)
        except StopIteration:
            break
        receipts += update.receipts
        if receipts:
            break
    gen.close()
    assert receipts, "stuck request was never terminated"
    assert receipts[0].outcome == "server unreachable"


def test_watch_terminates_request_on_log_rotation(tmp_path):
    """A replaced log (server restart) ends whatever was tracked."""
    path = tmp_path / "server.log"
    gen = _live_watch(
        path,
        poll_interval=0.01,
        ps_interval=60,
        query_server=False,
    )
    first = next(gen)
    assert first.state.phase is Phase.IDLE
    _append(path, *REQUEST_LINES)
    active = [next(gen) for _ in range(3)]
    assert active[-1].state.active
    path.rename(tmp_path / "server.log.old")
    write_log(path, ["fresh log after restart"])
    rotated, receipts = False, []
    for _ in range(500):
        update = next(gen)
        receipts += update.receipts
        if update.rotated:
            rotated = True
            break
    gen.close()
    assert rotated
    assert receipts and receipts[0].outcome == "log rotated"


def test_replay_boundary_resets_phantom_request(tmp_path):
    """History replay must not leave a phantom tracked into the live tail."""
    path = tmp_path / "server.log"
    write_log(
        path,
        [
            "slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 2050",
            "slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   1024, progress = 0.50, t =   7.78 s / 131.61 tokens per second",
        ],
    )
    gen = watch(
        log=str(path),
        replay=True,
        follow=True,
        poll_interval=0.01,
        ps_interval=60,
        query_server=False,
    )
    updates = [next(gen) for _ in range(6)]
    gen.close()
    assert updates[0].state.phase is Phase.PREFILL  # replayed history
    assert not any(u.receipts for u in updates)  # nothing completed
    assert updates[-1].state.phase is Phase.IDLE  # boundary reset the phantom


def test_live_request_after_replay_starts_fresh(tmp_path):
    """Lines past the boundary build a new request, not a supersede chain."""
    path = tmp_path / "server.log"
    write_log(
        path,
        [
            "slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 2050",
        ],
    )
    gen = watch(
        log=str(path),
        replay=True,
        follow=True,
        poll_interval=0.01,
        ps_interval=60,
        query_server=False,
    )
    for _ in range(4):  # drain replay lines + boundary
        next(gen)
    with open(path, "a") as handle:
        handle.write(
            "slot   operator(): id  0 | task 8 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 300\n"
        )
    fresh = [next(gen) for _ in range(6)]
    gen.close()
    assert not any(u.receipts for u in fresh)  # no phantom superseded
    assert fresh[-1].state.phase is Phase.PREFILL
    assert fresh[-1].state.prompt_tokens == 300


FINISHED_LLAMACPP = [
    "slot   operator(): id  0 | task 470 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 15",
    "slot init_sampler: id  0 | task 470 | init sampler, took 0.00 ms, tokens: text = 15, total = 15",
    "slot print_timing: id  0 | task 470 |        eval time =    3823.23 ms /    20 tokens (  201.22 ms per token,     4.97 tokens per second)",
    "slot      release: id  0 | task 470 | stop processing: n_tokens = 34, truncated = 0",
    "srv  update_slots: all slots are idle",
    '[GIN] 2026/09/06 - 17:39:12 | 200 |         1m57s |       127.0.0.1 | POST     "/v1/chat/completions"',
    '[GIN] 2026/09/06 - 17:39:17 | 200 |      82.083µs |       127.0.0.1 | GET      "/api/ps"',
]

FINISHED_MLX = [
    CACHE_HIT,
    PROGRESS_1,
    'time=2026-09-04T14:09:10.000+09:00 level=INFO source=server.go:1 msg="peak memory" size="28.02 GiB"',
    '[GIN] 2026/09/04 - 14:09:30 | 200 |         85s |       127.0.0.1 | POST     "/v1/chat/completions"',
    '[GIN] 2026/09/04 - 14:09:40 | 200 |      82.083µs |       127.0.0.1 | GET      "/api/ps"',
]


def test_finished_llamacpp_request_is_not_replayed(tmp_path):
    """A request that ended before we looked must not seed one as live.

    llama.cpp slot lines carry no clock, so replaying them reads as a request
    generating right now -- the dashboard showed `Generating` on an idle server.
    """
    log = write_log(tmp_path / "server.log", FINISHED_LLAMACPP)
    assert [line for line in follow_lines(str(log), follow=False)] == []


def test_finished_mlx_request_is_not_replayed(tmp_path):
    log = write_log(tmp_path / "server.log", FINISHED_MLX)
    assert [line for line in follow_lines(str(log), follow=False)] == []


def test_seed_stops_at_the_last_request_not_an_older_one(tmp_path):
    """Seeding scans back from the end, so a closed request hides the one
    before it rather than replaying whichever opened most recently."""
    log = write_log(tmp_path / "server.log", [CACHE_HIT, PROGRESS_1] + FINISHED_LLAMACPP)
    assert [line for line in follow_lines(str(log), follow=False)] == []


def test_watch_starts_idle_on_a_log_whose_last_request_finished(tmp_path):
    log = write_log(tmp_path / "server.log", FINISHED_LLAMACPP)
    gen = watch(
        log=str(log),
        replay=False,
        follow=True,
        poll_interval=0.01,
        ps_interval=60,
        query_server=False,
    )
    updates = [next(gen) for _ in range(3)]
    gen.close()
    assert all(u.state.phase is Phase.IDLE for u in updates)
    assert not any(u.receipts for u in updates)


def test_seeded_request_is_dropped_when_no_model_is_loaded(tmp_path):
    """A replayed request the log never closed died with an earlier server."""
    path = tmp_path / "server.log"
    write_log(path, REQUEST_LINES)
    gen = watch(
        log=str(path),
        host="127.0.0.1:1",  # nothing listens: no model is loaded
        replay=False,
        follow=True,
        poll_interval=0.01,
        ps_interval=60,
        query_server=True,
    )
    updates = [next(gen) for _ in range(6)]
    gen.close()
    assert updates[0].seeded and updates[0].state.active  # replayed history
    assert not any(u.receipts for u in updates)  # a phantom is not a request
    assert updates[-1].state.phase is Phase.IDLE


COMPLETE_LLAMACPP = [
    '[GIN] 2026/09/06 - 17:36:35 | 200 |      82.083µs |       127.0.0.1 | GET      "/api/ps"',
    "slot   operator(): id  0 | task 470 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 15",
    "slot init_sampler: id  0 | task 470 | init sampler, took 0.00 ms, tokens: text = 15, total = 15",
    "slot print_timing: id  0 | task 470 | prompt eval time =    1337.54 ms /    15 tokens (   89.17 ms per token,    11.21 tokens per second)",
    "slot print_timing: id  0 | task 470 |        eval time =    3823.23 ms /    20 tokens (  201.22 ms per token,     4.97 tokens per second)",
    "slot      release: id  0 | task 470 | stop processing: n_tokens = 34, truncated = 0",
    '[GIN] 2026/09/06 - 17:36:42 | 200 |          6.9s |       127.0.0.1 | POST     "/v1/chat/completions"',
]


def test_replay_dates_clockless_lines_by_the_log(tmp_path):
    """Replayed llama.cpp lines carry no clock; read at wall time they would
    imply a request that started when the replay did, matching no completion."""
    log = write_log(tmp_path / "server.log", COMPLETE_LLAMACPP)
    gen = watch(
        log=str(log),
        replay=True,
        follow=True,
        poll_interval=0.01,
        ps_interval=60,
        query_server=False,
    )
    receipts = []
    for _ in range(12):
        receipts += next(gen).receipts
        if receipts:
            break
    gen.close()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.outcome == "completed"
    assert receipt.status == "200"
    # the log's own clock, not the wall clock of the replay
    assert receipt.ts == pytest.approx(parse_gin_ts("2026/09/06", "17:36:42"))
    assert receipt.prefill_rate == pytest.approx(11.21)  # the runner's own figure
    assert receipt.decode_rate == pytest.approx(4.97)


def test_replay_boundary_releases_a_held_receipt(tmp_path):
    """A request completing on the last line is real, not a phantom: the
    receipt is held for trailing lines that the end of the log never brings."""
    log = write_log(tmp_path / "server.log", COMPLETE_LLAMACPP)
    updates = list(
        watch(log=str(log), replay=True, follow=False, query_server=False)
    )
    assert [r.status for u in updates for r in u.receipts] == ["200"]
