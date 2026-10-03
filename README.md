# Multi-Agent Channel

Specialized AI agents with real tools, persistent memory, and a token
budget that keeps long agent loops affordable. Runs on any model available
through [OpenRouter](https://openrouter.ai/).

## Highlights

- **Agent harness** — the loop, the tools, and the memory in one shell
- **Dynamic orchestration** — a triage call decides per request how many agents it deserves (simple → one, complex → up to 8)
- **Plan-and-execute** — big tasks split into steps, each run by a fresh subagent with a clean context window
- **Persistent memory** — a markdown wiki the agent writes itself, compounding across runs
- **Token budget proxy** — long inputs compressed, tool results truncated, history compacted
- **Cache-aware budgeting** — sticky compaction and session pinning keep prompt-cache prefixes hitting; `:cost` reports real spend per model
- **REPL sessions** — the harness remembers the conversation across turns; `:reset` starts fresh
- **MCP client** — tools from any Model Context Protocol server join the toolset over stdio, no extra dependencies
- **Reliability layer** — retries with backoff, model fallback cascade, optional reflection pass
- **Trace spans** — every run traced to `logs/spans.jsonl` with OpenTelemetry GenAI attribute names
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
and answers. The harness remembers the conversation between turns and
compacts it when it grows too large. Extra commands: `:plan <big task>`
splits it into steps run by fresh subagents, `:ingest <text>` files
knowledge into the memory wiki, `:lint` health-checks the wiki, `:cost`
prints a spend report per model, `:reset` starts a fresh conversation.

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
6. **Dynamic orchestration** (`orchestrator.py`) — before fanning out, a
   cheap triage call classifies the request: simple questions get one agent
   (no fan-out, no synthesis — 1 call instead of N+2); complex ones get a
   model-proposed agent count, hard-capped at `orchestrator.max_agents`.
   Triage and question generation use provider-native structured outputs
   when available (schema-constrained JSON), degrading to lenient parsing
   on providers without them.
7. **Plan-and-execute** (`planning.py`) — for big tasks (`:plan` in the
   harness): a planner emits a JSON step list; each step runs in a **fresh
   subagent with a clean context window** carrying only the task, a compact
   summary of prior steps, and its own instruction — long tasks never drown
   in a growing transcript. A final synthesis assembles the answer.
8. **MCP tools** (`mcp_client.py`) — servers listed under `mcp.servers` are
   launched over stdio, handshaked, and their tools join the built-in set
   (prefixed `mcp_<server>_<tool>`). A server that fails to start is
   skipped — MCP is strictly additive, stdlib only, no SDK required.
9. **Tracing** (`tracing.py`) — every run, LLM call, tool call, triage,
   worker, and plan step appends a span to `logs/spans.jsonl` with
   OpenTelemetry GenAI attribute names, one tree per request (workers nest
   under their orchestrate span even across threads). No daemon, no
   dependencies.

## Token cost & data safety

Agentic loops re-send the whole conversation on every iteration, so cost
compounds. The budget proxy caps that: inputs over ~4000 characters are
compressed extractively, oversized tool results are truncated, and history
older than the last few turns collapses into a digest once over budget.

Compaction is also **cache-aware**: history is rewritten in place only at
the moment it crosses the budget, so between compaction events the prompt
prefix is byte-stable and provider prompt caches keep hitting (cached
prefix reads are billed at a fraction of the input price). A session id is
pinned on every request so repeated calls land on the same provider
endpoint. Turned-on savings are visible in the model's
`cached_tokens` usage and in the `:cost` report, which sums real spend per
model from the local transcript.

**Nothing is lost.** What the model sees shrinks — what you keep doesn't:

- the local conversation always holds the full, untruncated history
- the system prompt, the original request, and recent turns are never compacted
- every original prompt and answer is written to `logs/transcript.jsonl`,
  so compressed inputs remain fully recoverable
- turns folded into a compaction digest are persisted verbatim first
  (`type: "compaction"` entries in the same file)

All knobs live under `harness:` in `config.yaml`.

## Configuration (`config.yaml`)

| Key | Default | Purpose |
|---|---|---|
| `openrouter.model` | — | Primary model, any OpenRouter ID |
| `openrouter.fallback_models` | `[]` | Tried in order if the primary keeps failing |
| `openrouter.request_timeout` | `120` | Per-request timeout (seconds) |
| `openrouter.session_id` | auto | Routing hint pinning requests to one provider endpoint (cache affinity) |
| `harness.history_budget` | `24000` | Est. token budget before history compaction |
| `harness.tool_result_limit` | `2000` | Max characters per tool result on the wire |
| `harness.input_compression_threshold` | `4000` | Inputs longer than this are compressed |
| `harness.retries.attempts` | `3` | Retries per model (backoff + jitter) |
| `harness.reflection.enabled` | `false` | Critic pass that re-answers flawed responses |
| `harness.transcript.enabled` | `true` | Log prompts/answers to `logs/transcript.jsonl` |
| `harness.tracing.enabled` | `true` | Append OTel GenAI spans to `logs/spans.jsonl` |
| `harness.planning.max_steps` | `10` | Cap on plan length for `:plan` tasks |
| `memory.enabled` | `true` | Persistent markdown-wiki memory |
| `mcp.servers` | `{}` | stdio MCP servers whose tools join the toolset |
| `orchestrator.dynamic` | `true` | Triage each request instead of a fixed agent count |
| `orchestrator.max_agents` | `8` | Hard cap on agents per request |
| `orchestrator.parallel_agents` | `4` | Agent count when dynamic is off, or triage fallback |

## Project layout

```
agent.py            # the agent loop (LLM + tools + memory)
harness.py          # user-facing shell: query / :plan / :cost / :reset / :ingest / :lint
orchestrator.py     # dynamic multi-agent fan-out + synthesis
planning.py         # plan-and-execute: steps as clean-window subagents
token_budget.py     # request compression proxy (cache-aware compaction)
reliability.py      # retries, backoff, model fallback
memory.py           # markdown-wiki memory store
mcp_client.py       # stdio MCP client: external tools, no SDK
tracing.py          # OTel GenAI spans -> logs/spans.jsonl
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
