"""Tests for the cache-aware token budget, session ids, usage/cost
tracking, and the :cost report."""

import json

import pytest

import token_budget as token_budget_module
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
  max_iterations: 3
harness:
  history_budget: 600
  tool_result_limit: 2000
  input_compression_threshold: 4000
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
    def __init__(self, message, usage=None):
        self.choices = [FakeChoice(message)]
        self.usage = usage


class FakeCompletions:
    """Captures every create() kwarg, including extra_body."""

    def __init__(self, responses):
        self.responses = list(responses)  # FakeResponse objects
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


class FakeToolCall:
    def __init__(self, name, arguments):
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


def wire_agent(agent, completions):
    agent.client = type("C", (), {})()
    agent.client.chat = type("Ch", (), {})()
    agent.client.chat.completions = completions
    return completions


# --------------------------------------------------------- sticky compaction

class TestStickyCompaction:
    def test_under_budget_is_noop(self):
        budget = TokenBudget({"history_budget": 10000})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        before = [dict(m) for m in messages]
        assert budget.compact_history(messages) == []
        assert messages == before

    def test_over_budget_rewrites_in_place(self):
        budget = TokenBudget({"history_budget": 400, "recent_turns_kept": 2})
        messages = [{"role": "system", "content": "s"}]
        messages.append({"role": "user", "content": "original request"})
        for i in range(10):
            messages.append({"role": "assistant", "content": f"answer {i} " + "z" * 200})
            messages.append({"role": "user", "content": f"next {i} " + "y" * 200})

        dropped = budget.compact_history(messages)

        assert dropped, "expected some turns to be folded into the digest"
        # Head preserved verbatim: system + original request
        assert messages[0]["content"] == "s"
        assert messages[1]["content"] == "original request"
        # A digest message sits between head and the verbatim tail
        assert "auto-compacted" in messages[2]["content"]
        # Everything folded away is returned so callers can persist it
        dropped_contents = " ".join(m.get("content") or "" for m in dropped)
        assert "answer 0" in dropped_contents
        # And the result fits the budget
        assert token_budget_module.estimate_messages_tokens(messages) <= 400

    def test_prefix_stable_between_compaction_events(self):
        """The cache-hit property: when no compaction happens this
        iteration, the prepared wire form only grows by appends, so the
        previous wire form is a byte-stable prefix of the next one."""
        budget = TokenBudget({"history_budget": 600, "recent_turns_kept": 4})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        prev_prepared = None
        compactions = 0

        for i in range(20):
            dropped = budget.compact_history(messages)
            prepared = budget.prepare_messages(messages)

            if dropped:
                compactions += 1
            elif prev_prepared is not None:
                # no compaction this round: pure append - cacheable prefix
                assert prepared[: len(prev_prepared)] == prev_prepared

            prev_prepared = prepared
            messages.append({"role": "assistant", "content": f"answer {i} " + "x" * 300})
            messages.append({"role": "user", "content": f"next {i} " + "y" * 300})

        assert compactions >= 1, "test should have crossed the budget at least once"


# ------------------------------------------------------------- session id

class TestSessionId:
    def test_extra_body_carries_session_id(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = wire_agent(agent, FakeCompletions([FakeMessage("ok")]))
        agent.call_llm([{"role": "user", "content": "hi"}])
        call = fake.calls[0]
        assert call["extra_body"]["session_id"] == agent.session_id

    def test_configured_session_id_wins(self, workdir):
        config_path = workdir / "config.yaml"
        config = config_path.read_text(encoding="utf-8").replace(
            'model: "test-model"', 'model: "test-model"\n  session_id: "fixed-session"'
        )
        config_path.write_text(config, encoding="utf-8")
        agent = OpenRouterAgent(silent=True)
        assert agent.session_id == "fixed-session"
        fake = wire_agent(agent, FakeCompletions([FakeMessage("ok")]))
        agent.call_llm([{"role": "user", "content": "hi"}])
        assert fake.calls[0]["extra_body"]["session_id"] == "fixed-session"


# --------------------------------------------------------- usage recording

class TestUsageRecording:
    def test_cached_tokens_and_cost_recorded(self, workdir):
        agent = OpenRouterAgent(silent=True)
        response = FakeResponse(FakeMessage("ok"), usage=FakeUsage(100, 20, cached_tokens=70, cost=0.012))
        agent._record_usage(response, "test-model")
        assert agent.get_usage() == {
            "prompt_tokens": 100, "completion_tokens": 20,
            "cached_tokens": 70, "cost": 0.012, "requests": 1,
        }

    def test_by_model_aggregation(self, workdir):
        agent = OpenRouterAgent(silent=True)
        agent._record_usage(FakeResponse(FakeMessage("a"), FakeUsage(100, 10)), "model-a")
        agent._record_usage(FakeResponse(FakeMessage("b"), FakeUsage(50, 5, cached_tokens=25)), "model-b")
        agent._record_usage(FakeResponse(FakeMessage("c"), FakeUsage(30, 3)), "model-a")
        by_model = agent.get_usage_by_model()
        assert by_model["model-a"]["prompt_tokens"] == 130
        assert by_model["model-a"]["requests"] == 2
        assert by_model["model-b"]["cached_tokens"] == 25

    def test_missing_details_default_to_zero(self, workdir):
        agent = OpenRouterAgent(silent=True)

        class BareUsage:
            prompt_tokens = 10
            completion_tokens = 2

        agent._record_usage(FakeResponse(FakeMessage("a"), BareUsage()), "m")
        usage = agent.get_usage()
        assert usage["cached_tokens"] == 0
        assert usage["cost"] == 0.0


# ------------------------------------------------------- transcript deltas

class TestTranscriptDeltas:
    def test_entries_record_per_run_deltas(self, workdir):
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, FakeCompletions([
            FakeResponse(FakeMessage("first"), FakeUsage(10, 2)),
            FakeResponse(FakeMessage("second"), FakeUsage(15, 3)),
        ]))

        agent.run("question one")
        agent.run("question two")

        entries = [json.loads(line)
                   for line in (workdir / "logs" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 2
        assert entries[0]["usage"]["prompt_tokens"] == 10
        assert entries[1]["usage"]["prompt_tokens"] == 15
        # Deltas sum to the agent's lifetime totals
        assert entries[0]["usage"]["requests"] + entries[1]["usage"]["requests"] == 2
        assert entries[0]["model"] == "test-model"
        assert "by_model" in entries[0]

    def test_compaction_persists_dropped_turns(self, workdir):
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, FakeCompletions([FakeMessage("ok")]))

        # SDK-shaped tool calls must serialize without crashing
        sdk_tool_call = FakeToolCall("search_web", '{"query": "cache"}')
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "thinking", "tool_calls": [sdk_tool_call]},
            {"role": "tool", "name": "search_web", "tool_call_id": "1", "content": "results"},
            {"role": "assistant", "content": "old answer 1 " + "z" * 900},
            {"role": "user", "content": "old question 2 " + "y" * 900},
            {"role": "assistant", "content": "old answer 3 " + "x" * 900},
            {"role": "user", "content": "latest question"},
        ]
        dropped_count = agent.compact_history(messages)

        assert dropped_count > 0
        entries = [json.loads(line)
                   for line in (workdir / "logs" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        compaction = [e for e in entries if e.get("type") == "compaction"]
        assert len(compaction) == 1
        dropped = compaction[0]["dropped"]
        # The dropped middle turns (here: the tool-call pair) are recoverable
        # verbatim - no data loss
        assert any(m.get("content") == "results" for m in dropped)
        assert any(m.get("tool_calls") == [{"name": "search_web", "arguments": '{"query": "cache"}'}]
                   for m in dropped)


# -------------------------------------------------------------- :cost report

class TestCostReport:
    def make_harness(self, workdir):
        return AgentHarness(silent=True)

    def test_aggregates_by_model(self, workdir):
        (workdir / "logs").mkdir()
        lines = [
            {"model": "m1", "input": "a", "output": "b", "usage": {"requests": 1},
             "by_model": {"m1": {"prompt_tokens": 100, "completion_tokens": 10,
                                 "cached_tokens": 60, "cost": 0.01, "requests": 1}}},
            {"model": "m1", "input": "c", "output": "d", "usage": {"requests": 1},
             "by_model": {"m1": {"prompt_tokens": 50, "completion_tokens": 5,
                                 "cached_tokens": 0, "cost": 0.005, "requests": 1},
                          "m2": {"prompt_tokens": 20, "completion_tokens": 2,
                                 "cached_tokens": 0, "cost": 0.002, "requests": 1}}},
            {"type": "compaction", "dropped": []},
        ]
        with (workdir / "logs" / "transcript.jsonl").open("w", encoding="utf-8") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")

        report = self.make_harness(workdir).cost_report()
        assert "m1" in report and "m2" in report
        assert "150" in report          # summed prompt tokens for m1
        assert "$0.0150" in report      # summed cost for m1
        assert "2 run(s)" in report     # compaction entries are not runs

    def test_legacy_entries_without_by_model(self, workdir):
        (workdir / "logs").mkdir()
        entry = {"model": "old-model", "input": "a", "output": "b",
                 "usage": {"prompt_tokens": 42, "completion_tokens": 7,
                           "cached_tokens": 0, "cost": 0.0, "requests": 1}}
        (workdir / "logs" / "transcript.jsonl").write_text(
            json.dumps(entry) + "\n", encoding="utf-8")
        report = self.make_harness(workdir).cost_report()
        assert "old-model" in report
        assert "42" in report

    def test_missing_transcript(self, workdir):
        report = self.make_harness(workdir).cost_report()
        assert "No transcript yet" in report
