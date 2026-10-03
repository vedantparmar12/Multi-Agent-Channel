"""Tests for the OTel GenAI trace spans (tracing.py and its wiring)."""

import json

import pytest

import tracing
import orchestrator
import planning as planning_module
from agent import OpenRouterAgent
from orchestrator import TaskOrchestrator
from planning import PlanExecutor


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Fresh directory per test and tracing back to its default state
    (a disabled run in one test must not silence the next)."""
    monkeypatch.chdir(tmp_path)
    tracing.configure({})
    yield
    tracing.configure({})


def spans():
    return tracing.read_spans()


def by_name(name):
    return [s for s in spans() if s["name"] == name]


# ------------------------------------------------------------- core module

class TestSpanBasics:
    def test_span_written_with_identity_and_duration(self):
        with tracing.span("unit"):
            pass
        written = spans()
        assert len(written) == 1
        s = written[0]
        assert s["name"] == "unit"
        assert s["trace_id"]
        assert s["span_id"]
        assert s["parent_id"] is None
        assert isinstance(s["duration_ms"], float)
        assert s["duration_ms"] >= 0

    def test_nested_spans_share_trace_and_chain_parents(self):
        with tracing.span("outer", attributes={"gen_ai.system": "openrouter"}):
            with tracing.span("inner"):
                pass
        # Spans are written when they close, innermost first
        inner, outer = spans()
        assert inner["trace_id"] == outer["trace_id"]
        assert inner["parent_id"] == outer["span_id"]
        assert outer["parent_id"] is None
        assert outer["attributes"]["gen_ai.system"] == "openrouter"

    def test_sibling_spans_have_separate_traces(self):
        with tracing.span("a"):
            pass
        with tracing.span("b"):
            pass
        first, second = spans()
        assert first["trace_id"] != second["trace_id"]

    def test_exception_records_error_and_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            with tracing.span("doomed"):
                raise ValueError("boom")
        s = spans()[0]
        assert s["attributes"]["error.type"] == "ValueError"
        assert "boom" in s["attributes"]["error.message"]

    def test_disabled_writes_nothing(self):
        tracing.configure({"enabled": False})
        with tracing.span("hidden"):
            pass
        assert spans() == []
        assert not tracing.is_enabled()

    def test_set_usage_attributes(self):
        usage = type("U", (), {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "prompt_tokens_details": type("D", (), {"cached_tokens": 7})(),
            "cost": 0.002,
        })()
        with tracing.span("chat") as s:
            tracing.set_usage_attributes(s, usage)
        attrs = spans()[0]["attributes"]
        assert attrs["gen_ai.usage.input_tokens"] == 11
        assert attrs["gen_ai.usage.output_tokens"] == 4
        assert attrs["gen_ai.usage.cached_tokens"] == 7
        assert attrs["gen_ai.usage.cost"] == 0.002

    def test_set_usage_attributes_tolerates_missing_extras(self):
        usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 2})()
        with tracing.span("chat") as s:
            tracing.set_usage_attributes(s, usage)
            tracing.set_usage_attributes(s, None)
        attrs = spans()[0]["attributes"]
        assert attrs["gen_ai.usage.input_tokens"] == 3
        assert "gen_ai.usage.cached_tokens" not in attrs
        assert "gen_ai.usage.cost" not in attrs


# ---------------------------------------------------------- agent wiring

class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message, usage=None):
        self.choices = [FakeChoice(message)]
        self.usage = usage


class FakeToolCall:
    def __init__(self, name, arguments):
        self.id = f"call_{name}"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeTokenDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=0, cost=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = FakeTokenDetails(cached_tokens)
        if cost is not None:
            self.cost = cost


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


def wire_agent(agent, completions):
    agent.client = type("C", (), {})()
    agent.client.chat = type("Ch", (), {})()
    agent.client.chat.completions = completions
    return completions


AGENT_CONFIG = """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "test-model"
system_prompt: "You are a test agent."
agent:
  max_iterations: 5
harness:
  history_budget: 10000
memory:
  enabled: false
"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(AGENT_CONFIG, encoding="utf-8")
    return tmp_path


class TestAgentSpans:
    def test_run_emits_nested_run_and_chat_spans(self, workdir):
        agent = OpenRouterAgent(silent=True)
        usage = FakeUsage(120, 30, cached_tokens=80, cost=0.001)
        wire_agent(agent, FakeCompletions([
            FakeResponse(FakeMessage("final answer"), usage=usage),
        ]))

        agent.run("hello")

        run_spans = by_name("agent.run")
        chat_spans = by_name("chat")
        assert len(run_spans) == 1 and len(chat_spans) == 1
        run_span, chat_span = run_spans[0], chat_spans[0]

        assert chat_span["parent_id"] == run_span["span_id"]
        assert chat_span["trace_id"] == run_span["trace_id"]

        assert run_span["attributes"]["gen_ai.operation.name"] == "agent_run"
        assert run_span["attributes"]["gen_ai.request.model"] == "test-model"
        assert run_span["attributes"]["gen_ai.usage.input_tokens"] == 120
        assert run_span["attributes"]["gen_ai.usage.output_tokens"] == 30
        assert run_span["attributes"]["gen_ai.usage.cached_tokens"] == 80

        assert chat_span["attributes"]["gen_ai.request.model"] == "test-model"
        assert chat_span["attributes"]["gen_ai.usage.input_tokens"] == 120
        assert chat_span["attributes"]["gen_ai.usage.cached_tokens"] == 80

    def test_tool_call_emits_tool_span(self, workdir):
        agent = OpenRouterAgent(silent=True)
        agent.tool_mapping["echo"] = lambda **kw: {"ok": True, **kw}
        wire_agent(agent, FakeCompletions([
            FakeResponse(FakeMessage("thinking", tool_calls=[
                FakeToolCall("echo", json.dumps({"value": 1})),
            ])),
            FakeResponse(FakeMessage("done"), usage=FakeUsage(10, 5)),
        ]))

        agent.run("use the echo tool")

        tool_spans = by_name("tool echo")
        assert len(tool_spans) == 1
        attrs = tool_spans[0]["attributes"]
        assert attrs["gen_ai.tool.name"] == "echo"
        assert attrs["gen_ai.operation.name"] == "execute_tool"

        # The tool span must sit inside the run's trace, under a chat span
        run_span = by_name("agent.run")[0]
        assert tool_spans[0]["trace_id"] == run_span["trace_id"]

    def test_failed_tool_records_error_attributes(self, workdir):
        def explode(**kw):
            raise RuntimeError("tool crashed")

        agent = OpenRouterAgent(silent=True)
        agent.tool_mapping["explode"] = explode
        wire_agent(agent, FakeCompletions([
            FakeResponse(FakeMessage("thinking", tool_calls=[
                FakeToolCall("explode", "{}"),
            ])),
            FakeResponse(FakeMessage("recovered"), usage=FakeUsage(10, 5)),
        ]))

        agent.run("make it explode")  # handle_tool_call swallows the error

        attrs = by_name("tool explode")[0]["attributes"]
        assert attrs["error.type"] == "RuntimeError"
        assert "tool crashed" in attrs["error.message"]

    def test_tracing_disabled_in_config_writes_nothing(self, workdir):
        config = AGENT_CONFIG.replace(
            "harness:\n",
            "harness:\n  tracing:\n    enabled: false\n",
        )
        (workdir / "config.yaml").write_text(config, encoding="utf-8")
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, FakeCompletions([
            FakeResponse(FakeMessage("answer"), usage=FakeUsage(5, 5)),
        ]))

        agent.run("quiet run")

        assert spans() == []


# ------------------------------------------- orchestrator + planning wiring

ORCH_CONFIG = """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "test-model"
system_prompt: "You are a test agent."
agent:
  max_iterations: 5
orchestrator:
  dynamic: true
  max_agents: 6
  parallel_agents: 4
  task_timeout: 30
  aggregation_strategy: "consensus"
  question_generation_prompt: "questions for {num_agents} agents about: {user_input}"
  synthesis_prompt: "synthesize {num_responses} responses: {agent_responses}"
harness:
  planning:
    max_steps: 3
memory:
  enabled: false
"""


class FakeAgent:
    queue = []
    runs = []
    created = 0

    def __init__(self, silent=False, **kwargs):
        self.silent = silent
        self.tools = []
        self.tool_mapping = {}

    def run(self, prompt, **kwargs):
        FakeAgent.runs.append(prompt)
        if not FakeAgent.queue:
            raise AssertionError(f"no scripted response left: {prompt[:80]}")
        response = FakeAgent.queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @classmethod
    def reset(cls, *script):
        cls.queue = list(script)
        cls.runs = []
        cls.created = 0


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(orchestrator, "OpenRouterAgent", FakeAgent)
    monkeypatch.setattr(planning_module, "OpenRouterAgent", FakeAgent)
    FakeAgent.reset()
    return FakeAgent


@pytest.fixture
def orch_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(ORCH_CONFIG, encoding="utf-8")
    return tmp_path


class TestOrchestratorSpans:
    def test_workers_nest_under_orchestrate_across_threads(self, orch_workdir, fake_agent):
        FakeAgent.reset(
            json.dumps({"complexity": "complex", "num_agents": 3, "reasoning": "survey"}),
            json.dumps(["q1", "q2", "q3"]),
            "answer 1", "answer 2", "answer 3",
            "synthesized answer",
        )
        orch = TaskOrchestrator(silent=True)

        result = orch.orchestrate("tell me everything")
        assert result == "synthesized answer"

        root = by_name("orchestrate")
        assert len(root) == 1
        root_span = root[0]

        worker_spans = [s for s in spans() if s["name"].startswith("worker ")]
        assert len(worker_spans) == 3

        # The whole request is one trace: triage, workers, and synthesis all
        # descend from the orchestrate span, even though workers ran in
        # ThreadPoolExecutor threads
        for s in spans():
            assert s["trace_id"] == root_span["trace_id"]

        for s in worker_spans:
            assert s["parent_id"] == root_span["span_id"]
            assert s["attributes"]["gen_ai.operation.name"] == "worker"

        triage = by_name("triage")
        assert len(triage) == 1
        assert triage[0]["parent_id"] == root_span["span_id"]

    def test_simple_request_single_worker_span(self, orch_workdir, fake_agent):
        FakeAgent.reset(
            json.dumps({"complexity": "simple", "num_agents": 1, "reasoning": "one fact"}),
            "direct answer",
        )
        orch = TaskOrchestrator(silent=True)

        orch.orchestrate("what is 2+2")

        assert len(by_name("orchestrate")) == 1
        assert len(by_name("worker 0")) == 1
        assert by_name("triage")[0]["attributes"]["gen_ai.operation.name"] == "triage"


class TestPlanningSpans:
    def test_plan_steps_and_synthesis_span_tree(self, orch_workdir, fake_agent):
        FakeAgent.reset(
            json.dumps([{"step": 1, "description": "first"},
                        {"step": 2, "description": "second"}]),
            "result 1", "result 2",
            "final synthesis",
        )
        executor = PlanExecutor(silent=True)

        outcome = executor.execute("do two things")
        assert outcome["final"] == "final synthesis"

        root = by_name("plan_execute")[0]
        names = {s["name"] for s in spans()}

        assert {"plan_execute", "planner", "plan_step 1", "plan_step 2", "synthesis"} <= names

        for s in spans():
            assert s["trace_id"] == root["trace_id"]

        for name in ("planner", "plan_step 1", "plan_step 2", "synthesis"):
            assert by_name(name)[0]["parent_id"] == root["span_id"]

        step_attrs = by_name("plan_step 2")[0]["attributes"]
        assert step_attrs["plan.step"] == 2
        assert step_attrs["plan.total_steps"] == 2
