# ollama-watch

Live input-processing and generation progress for a local Ollama server.

Ollama's HTTP API sends **nothing** between a request and its first token. Prompt
processing happens in that silence, and the token counters
(`prompt_eval_count`, `prompt_eval_duration`) only arrive in the *final*
response chunk. So no client — opencode, Claude Code, your own script — can show
you how far along a long prompt is. There is no progress on the wire to render.

The runner does log it. `ollama-watch` parses that log into events, folds them
into request state, and either renders it or hands it to you.

```
[qwen3.8:27b-mlx 22GB 100%GPU ctx32768 ∞] Processing input ████████░░░░░░  79.3% 38.9k/49.1k cached 34.8k 64 tok/s eta 2m39s
```

Finished requests scroll above the live line as one-line receipts:

```
✔ 13:40:37 200 2m50s | input 41.9k cached 31.9k processed 10.0k @ 67 tok/s (2m50s) | out ~557 @ ~24 tok/s (23.1s) spec 0.84
✖ 14:00:21 500 2m58s | input 49.1k processed 12.3k @ 80 tok/s (2m48s)
```

Pure standard library. No runtime dependencies.

## Install

```bash
uv tool install .          # or: pipx install .
```

Or run it straight from a clone:

```bash
PYTHONPATH=src python3 -m ollama_watch.cli
```

## Use

```bash
ollama-watch                       # follow live, one status line
ollama-watch -d                    # full-screen dashboard
ollama-watch --replay              # replay this log's history, then follow
ollama-watch --replay --no-follow  # history + summary, then exit
ollama-watch --log /path/to/server.log --host 127.0.0.1:11434
ollama-watch --no-server           # skip /api/ps (no model details in the header)
```

`Ctrl-C` prints a session summary: median input and output rates, total tokens,
cache hit share, and failure count.

Park it in a split pane next to whatever is talking to Ollama.

## Dashboard

`ollama-watch -d` gives a btop-style full-screen view. `q` quits.

```
 ollama-watch
 model   qwen3.8:27b-mlx  36GB  100% GPU  ctx32768<51.1k  ∞

 status  Processing input
         ███████████████████████████████████░░░░░░░░░  80.2%  41.0k/51.1k
         cached 34.8k  eta 1m47s  ttft so far 2m10s

 input   ▆▅▅▅▅▄▄▄▅▅▅▅▄█▃           55 tok/s now   25-89
 output  ▂▅▂▁▄▄█▃▂                  11 tok/s last  9-68

 memory  ███████████████████████████░░░┊░  31.7 of 36 GiB   peak 34.9 (97%)

 session 1h23m   working 24m17s (29%)   idle 58m57s (71%)
         █████████▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░  █ input 18m51s  ▓ output 2m30s  ▒ other 2m56s  ░ idle 58m57s

       time      code    input   in/s   cache     out   out/s   queue    ttft     dur       peak
 + 13:42:22       200    50.3k     30     98%    ~303     ~12    2.0s   43.0s   1m08s   33.2 GiB
 ! 13:37:57       500    49.1k    164                            3.0s   5m03s   5m03s   32.6 GiB
 + 14:09:19       200     6930     46           ~1192     ~48    9.0s   2m40s   3m05s   35.0 GiB

 7 req  cache 68%  median in 46 tok/s  median out 22 tok/s  median ttft 2m56s  failed 1  q quit
```

## Session accounting

The `session` panel splits the wall clock into where it actually went: time the
model worked, versus time it sat loaded waiting for you to read, type or
approve something. Working time splits again into processing input, generating
output, and the remainder of a request that is neither (queueing, tokenising,
templating). Segments use distinct glyphs, not colours.

## Time to first token

Gin times the whole HTTP handler, so a request's arrival is `end - duration`.
From that: `queue` is arrival until the runner started work, and `ttft` is
arrival until input processing finished -- the wait before the first token can
appear. While a request is in flight the live panel shows `ttft so far`, which
is a lower bound, since the figure is only final once the request ends.

## Sparklines are history

One bar per **finished** request, **oldest on the left, newest on the right**.
The generation row has fewer bars than the input row: a request that failed or
was cancelled processed input but produced no output, so it contributes a bar
to one row and not the other.
They do not move during a request. The current figure sits immediately to the
right of the bars, beside the newest one, so the row reads oldest to newest to
now in one direction. The number
beside each says which figure it is: input reads `now` while input is being
processed, `req` for the current request once it moves to generating, and
`last` when idle. Generation always reads `last` -- nothing is logged per
token, so there is no live generation rate to show. When a request is in
flight, its provisional value appears after a `┃` divider so it ticks visibly
without being mistaken for a completed one.

The figure to the right of each sparkline is the min-max rate across its bars.
Sparklines are padded to a fixed width so that figure stays in one column
even though the two rows rarely hold the same number of bars.

Status is carried by a glyph (`+` ok, `!` failed, `-` other) and by fixed
columns, never by colour alone -- a monochrome terminal theme renders red and
green identically, so hue is an accent here and nothing depends on it.

Sparklines are per-request rates over recent history, each scaled to its own
peak. The memory gauge compares peak request memory against physical RAM and
turns yellow past 75% and red past 90% -- on a large prompt the KV cache, not
the weights, is what fills it.

It is `curses` from the standard library, drawn straight off `watch()`, which
yields an idle beat every poll interval. No second loop, no threads, no
dependencies.

## As a library

```python
from ollama_watch import Phase, watch

for update in watch():
    state = update.state
    if state.phase is Phase.PREFILL:
        print(f"{state.fraction:.0%} of {state.prompt_tokens}, eta {state.eta_s:.0f}s")
    for receipt in update.receipts:
        print(receipt.status, receipt.prompt_tokens, receipt.decode_rate)
```

The pieces are separable, and each layer is usable on its own:

| Module | Role |
|---|---|
| `parse.py` | Pure text → `Event`. No state, no I/O. |
| `events.py` | `CacheVerdict`, `Progress`, `RequestEnd`, `SpeculativeStats`, `SlotTiming`, … |
| `state.py` | `Tracker` folds events into `RequestState` and emits `Receipt`s. |
| `tail.py` | Follows the log: seeding, rotation, replay. |
| `client.py` | Read-only `/api/ps` lookup. |
| `render.py` | Terminal formatting. Touches neither log nor server. |
| `watch.py` | `watch()` — the whole pipeline as a generator of `Update`. |

Feeding a `Tracker` by hand takes three lines, which is also how the tests work:

```python
from ollama_watch import Tracker, parse_line

tracker = Tracker()
for line in open("server.log"):
    event = parse_line(line)
    if event:
        for receipt in tracker.feed(event):
            print(receipt)
```

## What the log actually exposes

Four runner behaviours shape this library. Each was found the hard way, and
each is pinned by a test.

**Progress is relative, not absolute.** A request opens with a prefix-cache
verdict:

```
msg="cache hit" total=49091 matched=34816 cached=34816 left=14275
```

Subsequent progress lines count only *new* tokens and their `total` is the
**remaining** count, not the prompt length:

```
msg="Prompt processing progress" processed=8192 total=14275
```

So absolute progress is `cached + processed` out of the verdict's `total`.
Reading the progress line alone under-reports by the whole cached prefix.

**The last token is never prefilled.** The runner processes every token but the
final one — that one is fed as generation begins — so `processed` stops one
short of `total` and an equality check for "done" never fires.

**Requests are concurrent.** A completion line only belongs to the tracked
request if its own reported duration lines up with the observed start.
Otherwise a 63 ms keep-alive ping "completes" a three-minute request.

**Trailing events straddle the end.** `peak memory` and the decode metrics can
arrive on either side of the request-end line -- the Gin access line carries
only second resolution, so sub-second ordering decides. A finished request is
briefly held so late arrivals attach, and metrics that arrive early are
buffered on the live request. Handling only the late case silently loses the
generation figures for most requests.

## Generation rate

Nothing is logged per generated token, so there is no live generation
throughput to show — the live line reports elapsed time and the previous
request's rate as a reference. Exact figures arrive when a request ends, from
whichever engine served it:

- **MLX** logs `speculative decode stats` (`iterations`, `accepted`,
  `acceptance`). Tokens produced is `accepted + iterations`: each iteration
  emits the accepted draft tokens plus one from the target model. Marked `~`
  because it is derived.
- **llama.cpp** logs `slot print_timing`, which states the generation rate
  outright. Reported exactly.

## Context length is a declaration, not a limit

`/api/ps` reports `context_length` — a VRAM-based default. The MLX engine grows
its KV cache past it rather than truncating: prompts of 63,307 tokens are
served in full against a declared 32,768, with peak memory climbing from
21.8 GiB to 36.1 GiB as it goes. When the prompt exceeds the declared window
the header says `ctx32768<63.3k` rather than implying a limit that is not being
enforced. Watch memory, not the number.

## Tests

```bash
uv run --with pytest python -m pytest
```
