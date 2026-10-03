"""Tests for provider-native structured outputs (response_format)."""

import json

import pytest

import orchestrator as orchestrator_module
import planning as planning_module
from agent import OpenRouterAgent, structured_output
from orchestrator import TaskOrchestrator
from planning import PlanExecutor


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "test-model"
system_prompt: "You are a test agent."
agent:
  max_iterations: 2
harness:
  retries:
    attempts: 1
  planning:
    max_steps: 5
memory:
  enabled: false
orchestrator:
  dynamic: true
  max_agents: 6
  parallel_agents: 4
  task_timeout: 30
  aggregation_strategy: "consensus"
  question_generation_prompt: "questions for {num_agents} about: {user_input}"
  synthesis_prompt: "synthesize {num_responses}: {agent_responses}"
""",
        encoding="utf-8",
    )
    return tmp_path


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


class RejectingThenPlainCompletions:
    """Rejects any request carrying response_format; serves others."""

    def __init__(self, plain_response):
        self.plain_response = plain_response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format") is not None:
            raise Exception("400: response_format json_schema is not supported by this provider")
        return FakeResponse(FakeMessage(self.plain_response))


class AlwaysFailingCompletions:
    def __init__(self, message):
        self.message = message

    def create(self, **kwargs):
        raise Exception(self.message)


def wire_agent(agent, completions):
    agent.client = type("C", (), {})()
    agent.client.chat = type("Ch", (), {})()
    agent.client.chat.completions = completions
    return completions


class TestStructuredOutputs:
    def test_call_llm_passes_response_format(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = RejectingThenPlainCompletions('{"ok": true}')
        wire_agent(agent, fake)
        fmt = structured_output("triage", {"type": "object"})
        agent.call_llm([{"role": "user", "content": "hi"}], response_format=fmt)
        assert fake.calls[0]["response_format"] == fmt
        assert fake.calls[0]["extra_body"]["session_id"] == agent.session_id

    def test_run_plumbs_format_through_loop(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = RejectingThenPlainCompletions('{"complexity": "simple"}')
        wire_agent(agent, fake)
        fmt = structured_output("triage", {"type": "object"})
        agent.run("classify this", response_format=fmt)
        assert fake.calls[0]["response_format"] == fmt

    def test_provider_rejection_degrades_to_plain(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = RejectingThenPlainCompletions('{"complexity": "simple", "num_agents": 1}')
        wire_agent(agent, fake)
        result = agent.run(
            "classify this",
            response_format=structured_output("triage", {"type": "object"}),
        )
        assert "simple" in result
        # exactly one structured attempt and one plain retry
        assert len(fake.calls) == 2
        assert "response_format" in fake.calls[0]
        assert "response_format" not in fake.calls[1]

    def test_unrelated_errors_are_not_swallowed(self, workdir):
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, AlwaysFailingCompletions("rate limit exceeded, try again"))
        with pytest.raises(Exception, match="rate limit"):
            agent.run(
                "classify this",
                response_format=structured_output("triage", {"type": "object"}),
            )

    def test_structured_output_helper_shape(self):
        fmt = structured_output("plan", {"type": "array"})
        assert fmt == {"type": "json_schema", "json_schema": {"name": "plan", "schema": {"type": "array"}}}


class RecordingAgent:
    """Fake OpenRouterAgent that records response_format kwargs."""

    formats = []

    def __init__(self, silent=False, **kwargs):
        self.silent = silent
        self.tools = []
        self.tool_mapping = {}

    def run(self, prompt, response_format=None, **kwargs):
        RecordingAgent.formats.append(response_format)
        if response_format is not None and "questions" in str(response_format):
            return json.dumps(["q1", "q2"])
        if response_format is not None and "plan" in str(response_format):
            return json.dumps([{"step": 1, "description": "only step"}])
        return json.dumps({"complexity": "simple", "num_agents": 1, "reasoning": ""})


@pytest.fixture
def recording_agent(monkeypatch):
    RecordingAgent.formats = []
    monkeypatch.setattr(orchestrator_module, "OpenRouterAgent", RecordingAgent)
    monkeypatch.setattr(planning_module, "OpenRouterAgent", RecordingAgent)
    return RecordingAgent


class TestCallersUseStructuredOutputs:
    def test_triage_uses_json_schema_format(self, workdir, recording_agent):
        orch = TaskOrchestrator(silent=True)
        triage = orch.triage_request("what time is it in tokyo?")
        assert triage["complexity"] == "simple"
        fmt = recording_agent.formats[0]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "triage"
        assert "complexity" in fmt["json_schema"]["schema"]["properties"]

    def test_decompose_uses_json_schema_format(self, workdir, recording_agent):
        orch = TaskOrchestrator(silent=True)
        questions = orch.decompose_task("survey the field", 2)
        assert questions == ["q1", "q2"]
        fmt = recording_agent.formats[0]
        assert fmt["json_schema"]["name"] == "questions"
        assert fmt["json_schema"]["schema"]["type"] == "array"

    def test_planner_uses_json_schema_format(self, workdir, recording_agent):
        executor = PlanExecutor(silent=True)
        plan = executor.create_plan("do a big thing")
        assert plan == [{"step": 1, "description": "only step"}]
        fmt = recording_agent.formats[0]
        assert fmt["json_schema"]["name"] == "plan"
        assert fmt["json_schema"]["schema"]["items"]["required"] == ["description"]
