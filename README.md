# Multi-Agent Channel

Specialized AI agents with real tools, persistent memory, and a token
budget that keeps long agent loops affordable. Runs on any model available
through [OpenRouter](https://openrouter.ai/).

## Highlights

- **Agent harness** — the loop, the tools, and the memory in one shell
- **Multi-agent orchestration** — parallel agents with AI task decomposition and a synthesis pass
- **Persistent memory** — a markdown wiki the agent writes itself, compounding across runs
- **Token budget proxy** — long inputs compressed, tool results truncated, history compacted
- **Reliability layer** — retries with backoff, model fallback cascade, optional reflection pass
- **Sandboxed tools** — web search, file read/write, calculator, code validation, PRPs

## Quick start

```bash
git clone https://github.com/vedantparmar12/Multi-Agent-Channel.git
cd Multi-Agent-Channel
pip install -r requirements.txt

cp config.yaml.example config.yaml
# edit config.yaml and add your OpenRouter API key
```

```bash
python harness.py
```

That's it. Type a question and the agent searches, reads files, calculates,
and answers. Two extra commands: `:ingest <text>` files knowledge into its
memory wiki, `:lint` health-checks the wiki.

## Entry points

| Command | What it does |
|---|---|
| `python harness.py` | One capable agent: loop + tools + memory (recommended) |
| `python main.py` | Minimal single-agent chat |
| `python make_it_heavy.py` | Fans out N parallel agents, then synthesizes one answer |

## How it works

An agent is a loop over an LLM, a set of tools, and notes it maintains
itself. This project is those three things plus a cost guard:

1. **The loop** — the model thinks, calls tools, observes results, repeats,
   then answers. Responses without tool calls end the loop; the
   `mark_task_complete` tool lets the model finish explicitly.
2. **The tools** — auto-discovered from `tools/`. Drop in a new file,
   subclass `BaseTool`, and the agent can use it on the next start. File
   access is sandboxed to the project root; credential files are blocked.
3. **The memory** — a markdown wiki in `memory/`: topic pages the agent
   writes via `save_memory_page`, an `index.md` injected into every system
   prompt, and a grep-able `log.md` timeline (`grep "^## \[" memory/log.md`).
   Knowledge compounds across runs instead of vanishing into chat history.
4. **The token budget** — every request passes through a local proxy
   (`token_budget.py`) before hitting the API, so loop iterations stop
   compounding input cost (details below).
5. **The reliability layer** — transient API errors retry with exponential
   backoff and jitter; if the primary model keeps failing, the configured
   fallbacks take over; an optional reflection pass has a critic re-answer
   flawed responses.

## Token cost & data safety

Agentic loops re-send the whole conversation on every iteration, so cost
compounds. The budget proxy caps that: inputs over ~4000 characters are
compressed extractively, oversized tool results are truncated, and history
older than the last few turns collapses into a digest once over budget.

**Nothing is lost.** What the model sees shrinks — what you keep doesn't:

- the local conversation always holds the full, untruncated history
- the system prompt, the original request, and recent turns are never compacted
- every original prompt and answer is written to `logs/transcript.jsonl`,
  so compressed inputs remain fully recoverable

All knobs live under `harness:` in `config.yaml`.

## Configuration (`config.yaml`)

| Key | Default | Purpose |
|---|---|---|
| `openrouter.model` | — | Primary model, any OpenRouter ID |
| `openrouter.fallback_models` | `[]` | Tried in order if the primary keeps failing |
| `openrouter.request_timeout` | `120` | Per-request timeout (seconds) |
| `harness.history_budget` | `24000` | Est. token budget before history compaction |
| `harness.tool_result_limit` | `2000` | Max characters per tool result on the wire |
| `harness.input_compression_threshold` | `4000` | Inputs longer than this are compressed |
| `harness.retries.attempts` | `3` | Retries per model (backoff + jitter) |
| `harness.reflection.enabled` | `false` | Critic pass that re-answers flawed responses |
| `harness.transcript.enabled` | `true` | Log prompts/answers to `logs/transcript.jsonl` |
| `memory.enabled` | `true` | Persistent markdown-wiki memory |
| `orchestrator.parallel_agents` | `4` | Number of agents fanned out by the orchestrator |

## Project layout

```
agent.py            # the agent loop (LLM + tools + memory)
harness.py          # user-facing shell: query / :ingest / :lint
orchestrator.py     # parallel multi-agent execution + synthesis
token_budget.py     # request compression proxy
reliability.py      # retries, backoff, model fallback
memory.py           # markdown-wiki memory store
tools/              # auto-discovered tool library
context/            # project-context loading + PRP engine
tests/              # pytest suite
```

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
