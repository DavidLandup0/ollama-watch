<div align="center">
<h3>ollama-watch</h3>
<p><i>Terminal observability for local model status via Ollama</i></p>
</div>

<hr>

<div align="center">
<p><strong>Live prefill and decode progress from Ollama's runner.</strong></p>
<p><strong>Standard library only. No dependencies.</strong></p>
</div>

<div align="center">
<a href="#dashboard">Dashboard</a> |
<a href="#as-a-library">Library</a> |
</div>
<hr>

## Quickstart

```bash
uv tool install .          # or: pipx install .
```

Run it:

```bash
ollama-watch
```

Watch a long prompt go through:

```
[qwen3.8:27b-mlx 22GB 100%GPU ctx32768] Processing input ████████░░░░░░  79.3% 38.9k/49.1k cached 34.8k 64 tok/s eta 2m39s
```

Finished requests scroll above the live line as one-line receipts:

```
✔ 13:40:37 200 2m50s | input 41.9k cached 31.9k processed 10.0k @ 67 tok/s (2m50s) | out ~557 @ ~24 tok/s (23.1s) mtp_accept_rate 0.84
✖ 14:00:21 500 2m58s | input 49.1k processed 12.3k @ 80 tok/s (2m48s)
```

`Ctrl+C` prints a session summary: median input and output rates, total tokens, cache hit share, and failure count.

Or run it straight from a clone:

```bash
PYTHONPATH=src python3 -m ollama_watch.cli
```

### Options

```bash
ollama-watch                       # follow live, one status line
ollama-watch -d                    # full-screen dashboard
ollama-watch --replay              # replay this log's history, then follow
ollama-watch --replay --no-follow  # history + summary, then exit
ollama-watch --log /path/to/server.log --host 127.0.0.1:11434
ollama-watch --no-server           # skip /api/ps (no model details in the header)
```

## Dashboard

`ollama-watch -d` gives a full-screen view. `q` quits.

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

### Session Accounting

The `session` panel splits the wall clock into where it was spent: model-work time, idle time, etc.

Working time splits again into processing input, generating output, and the remainder of a request that is neither (queueing, tokenising, templating).

### Time To First Token

Ollama's Gin times the whole HTTP handler, so a request's arrival is `end - duration`.

From that: `queue` is arrival until the runner started work, and `ttft` is arrival until input processing finished - the wait before the first token can appear. While a request is in flight the live panel shows `ttft so far`, which is a lower bound, since the figure is only final once the request ends.

### Sparklines

Sparklines display recent history. One bar per **finished** request, **oldest on the left, newest on the right**.

The figure to the right of each sparkline is the min-max rate across its bars. Sparklines are padded to a fixed width so that figure stays in one column even though the two rows rarely hold the same number of bars.

Also, each sparkline is scaled to its own peak.

## As A Library

Everything the CLI shows is available programmatically, so you can build your own CLI, your own dashboard, or feed the numbers into something else entirely — a status bar, a progress widget in your own Ollama client, a metrics exporter, etc.

`watch()` is the whole pipeline as a generator. It yields an `Update` whenever anything changes:

```python
from ollama_watch import Phase, watch

for update in watch():
    state = update.state
    if state.phase is Phase.PREFILL:
        print(f"{state.fraction:.0%} of {state.prompt_tokens}, eta {state.eta_s:.0f}s")
    for receipt in update.receipts:
        print(receipt.status, receipt.prompt_tokens, receipt.decode_rate)
```

A twenty-line progress bar, which is roughly what `ollama-watch` without the dashboard is:

```python
from ollama_watch import Phase, watch

last = None
for update in watch():
    state = update.state
    if state.phase is Phase.PREFILL and state.fraction != last:
        last = state.fraction
        filled = int(state.fraction * 40)
        eta = f"eta {state.eta_s:.0f}s" if state.eta_s else ""
        print(f"\r[{'#' * filled:<40}] {state.fraction:.0%} {eta}", end="", flush=True)
    for receipt in update.receipts:
        last = None
        mark = "ok" if receipt.ok else receipt.status
        ttft = f"ttft {receipt.ttft_s:.0f}s" if receipt.ttft_s else "ttft unknown"
        print(f"\r{mark} {receipt.prompt_tokens} in, {ttft}")
```

## Tests

```bash
uv run --with pytest python -m pytest
```

## Contributing

Found a bug? Have an idea? PRs welcome.

1. Fork it
2. Create your feature branch
3. Write tests
4. Submit a PR
