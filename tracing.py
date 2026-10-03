"""Hierarchical trace spans with OpenTelemetry GenAI attribute names.

Every run produces a trace: spans for the whole run, each LLM call,
triage, each parallel worker, plan steps, and tool calls. Spans are
appended to logs/spans.jsonl as they close - no daemon, no dependencies
- with attribute names matching the OTel GenAI semantic conventions
(``gen_ai.system``, ``gen_ai.operation.name``, ``gen_ai.request.model``,
``gen_ai.usage.input_tokens``, ``gen_ai.tool.name``, ``error.type``), so
the log drops into OTel-compatible tooling unchanged.

Span parenting uses contextvars. ThreadPoolExecutor copies the caller's
context at submit time only on Python 3.14+, so the orchestrator
explicitly snapshots a context per worker it submits - workers attach to
the "orchestrate" span on every supported Python version, giving one
tree per request even across threads.

API:

    import tracing

    with tracing.span("triage", attributes={"gen_ai.system": "openrouter"}) as s:
        s.set("gen_ai.request.model", "test-model")

Disable with ``harness.tracing.enabled: false``.
"""

import contextlib
import json
import threading
import time
import uuid
import contextvars
from pathlib import Path
from typing import Dict, Optional

_SPANS_PATH = Path("logs") / "spans.jsonl"
_WRITE_LOCK = threading.Lock()
_current_span = contextvars.ContextVar("mac_current_span", default=None)
_enabled = True


def configure(config: dict) -> None:
    """Apply the harness.tracing config section (enabled, default true)."""
    global _enabled
    _enabled = bool((config or {}).get("enabled", True))


def is_enabled() -> bool:
    return _enabled


def read_spans(path=None) -> list:
    """Load spans.jsonl (tests and tooling)."""
    path = Path(path) if path else _SPANS_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Span:
    """One span: identity, timing, and OTel-conventioned attributes."""

    def __init__(self, name: str, attributes: Optional[Dict], trace_id: str, parent_id: Optional[str]):
        self.data = {
            "trace_id": trace_id,
            "span_id": uuid.uuid4().hex[:16],
            "parent_id": parent_id,
            "name": name,
            "start": round(time.time(), 3),
            "attributes": dict(attributes or {}),
        }

    def set(self, key: str, value) -> None:
        self.data["attributes"][key] = value


@contextlib.contextmanager
def span(name: str, attributes: Optional[Dict] = None):
    """Open a span; it is written to logs/spans.jsonl when the block exits."""
    parent = _current_span.get()
    trace_id = parent.data["trace_id"] if parent else uuid.uuid4().hex[:16]
    current = Span(name, attributes, trace_id, parent.data["span_id"] if parent else None)
    token = _current_span.set(current)
    started = time.perf_counter()
    try:
        yield current
    except Exception as e:
        current.set("error.type", type(e).__name__)
        current.set("error.message", str(e)[:300])
        raise
    finally:
        _current_span.reset(token)
        current.data["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if _enabled:
            _write(current.data)


def set_usage_attributes(span_obj: "Span", usage) -> None:
    """Copy gen_ai.usage.* attributes from an API usage object, if present.

    Tolerates missing/None usage and OpenAI-style usage objects that lack
    the OpenRouter extras (cached_tokens, cost).
    """
    if not usage:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    span_obj.set("gen_ai.usage.input_tokens", prompt)
    span_obj.set("gen_ai.usage.output_tokens", completion)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    if cached:
        span_obj.set("gen_ai.usage.cached_tokens", cached)
    cost = getattr(usage, "cost", None)
    if cost:
        span_obj.set("gen_ai.usage.cost", cost)


def _write(data: dict) -> None:
    # Tracing must never fail the run it observes
    try:
        _SPANS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, _SPANS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass
