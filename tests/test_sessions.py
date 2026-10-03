"""Tests for the rolling REPL session (history + between-turn compaction)."""

import pytest

from token_budget import TokenBudget
from agent import OpenRouterAgent
from harness import AgentHarness


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
  history_budget: 800
  recent_turns_kept: 4
memory:
  enabled: false
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
    def __init__(self, message):
        self.choices = [FakeChoice(message)]
        self.usage = None


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSessionAgent:
    """Stands in for harness.agent; records run() history arguments."""

    def __init__(self):
        self.runs = []  # (prompt, history)
        self.compacted = []
        self.budget = TokenBudget({"history_budget": 100000})
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "cost": 0.0, "requests": 0}
        self.mcp_clients = []

    def run(self, prompt, response_format=None, history=None):
        # Record a copy: the harness keeps appending to the session list
        # it passes, and assertions must see it as it was at call time
        self.runs.append((prompt, list(history) if history is not None else None))
        return f"answer to: {prompt}"

    def get_usage(self):
        return dict(self.usage)

    def compact_history(self, messages):
        self.compacted.append(list(messages))
        return 0

    def close(self):
        pass


def make_harness(workdir):
    harness = AgentHarness(silent=True)
    harness.agent = FakeSessionAgent()
    harness.memory_store = None
    return harness


class TestHarnessSession:
    def test_history_replayed_on_followup(self, workdir):
        harness = make_harness(workdir)
        harness.query("what is OpenRouter?")
        harness.query("why does it matter?")

        first_prompt, first_history = harness.agent.runs[0]
        second_prompt, second_history = harness.agent.runs[1]
        assert first_history == []
        # The second query sees the first turn pair as history
        assert second_history == [
            {"role": "user", "content": "what is OpenRouter?"},
            {"role": "assistant", "content": "answer to: what is OpenRouter?"},
        ]

    def test_session_accumulates_turns(self, workdir):
        harness = make_harness(workdir)
        harness.query("one")
        harness.query("two")
        assert len(harness.session) == 4
        assert harness.session[2]["content"] == "two"

    def test_session_compacted_between_turns(self, workdir):
        harness = make_harness(workdir)
        harness.query("one")
        # After each query the harness bounds the session via the agent
        assert len(harness.agent.compacted) == 1
        assert harness.agent.compacted[0] == harness.session

    def test_reset_clears_session(self, workdir):
        harness = make_harness(workdir)
        harness.query("remember this")
        harness.reset_session()
        assert harness.session == []
        harness.query("fresh start")
        assert harness.agent.runs[1][1] == []

    def test_ingest_and_lint_do_not_join_the_session(self, workdir):
        harness = make_harness(workdir)
        harness.agent.runs.clear()
        harness.ingest("some source text")
        assert harness.session == []
        # ingest ran without history
        assert harness.agent.runs[0][1] is None


class TestAgentHistory:
    def wire(self, agent, responses):
        fake = FakeCompletions(responses)
        agent.client = type("C", (), {})()
        agent.client.chat = type("Ch", (), {})()
        agent.client.chat.completions = fake
        return fake

    def test_history_prepended_to_messages(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = self.wire(agent, [FakeResponse(FakeMessage("ok"))])
        history = [
            {"role": "user", "content": "what is the budget?"},
            {"role": "assistant", "content": "7 million dollars"},
        ]
        agent.run("and the deadline?", history=history)

        sent = fake.calls[0]["messages"]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
        assert sent[1]["content"] == "what is the budget?"
        assert sent[2]["content"] == "7 million dollars"
        assert sent[3]["content"] == "and the deadline?"

    def test_long_history_compacted_on_the_wire(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = self.wire(agent, [FakeResponse(FakeMessage("ok"))])
        # Three Q&A pairs of ~600 chars each: the replayed history crosses
        # history_budget=800 but the verbatim tail still fits after the
        # digest folds the middle away
        history = []
        for i in range(3):
            history.append({"role": "user", "content": f"question {i} " + "q" * 600})
            history.append({"role": "assistant", "content": f"old turn {i} " + "x" * 600})
        agent.run("latest question", history=history)

        sent = fake.calls[0]["messages"]
        assert any("auto-compacted" in (m.get("content") or "") for m in sent)
        assert sent[-1]["content"] == "latest question"
        # ...and the dropped turns were persisted to the transcript
        import json
        entries = [json.loads(line) for line in
                   (workdir / "logs" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        compaction = [e for e in entries if e.get("type") == "compaction"]
        assert compaction
        assert "old turn 0" in compaction[0]["dropped"][0]["content"]
