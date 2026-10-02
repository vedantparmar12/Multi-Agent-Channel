"""Tests for the reliability layer: retries with backoff, model fallback
cascade, reflection quality mode, and the no-data-loss transcript."""

import json

import pytest

from reliability import is_transient_error, retry_call
from agent import OpenRouterAgent
from harness import AgentHarness


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated directory with a reliability-focused config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "primary-model"
  fallback_models: ["backup-model"]
system_prompt: "You are a test agent."
agent:
  max_iterations: 5
harness:
  retries:
    attempts: 2
    base_delay: 0.001
  transcript:
    enabled: true
  input_compression_threshold: 100
  input_compression_ratio: 0.5
memory:
  enabled: true
  directory: memory
""",
        encoding="utf-8",
    )
    return tmp_path


def enable_reflection(workdir):
    config_path = workdir / "config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace(
        "  transcript:",
        "  reflection:\n    enabled: true\n  transcript:",
    )
    config_path.write_text(config, encoding="utf-8")


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


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


class CascadeCompletions:
    """Fake API that fails for the given models, tracking model usage."""

    def __init__(self, fail_models, content="ok", usage=None):
        self.fail_models = set(fail_models)
        self.content = content
        self.usage = usage
        self.models_used = []
        self.last_messages = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.models_used.append(model)
        self.last_messages.append(kwargs["messages"])
        if model in self.fail_models:
            raise Exception("rate limit exceeded (429)")
        return FakeResponse(FakeMessage(self.content), self.usage)


def wire_agent(agent, completions):
    agent.client = type("C", (), {})()
    agent.client.chat = type("Ch", (), {})()
    agent.client.chat.completions = completions
    return completions


# ----------------------------------------------------------------- retries

class TestRetryCall:
    def test_retries_transient_then_succeeds(self):
        sleeps = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("503 service unavailable")
            return "recovered"

        result = retry_call(flaky, attempts=5, base_delay=1.0, sleep=sleeps.append)
        assert result == "recovered"
        assert len(sleeps) == 2
        # backoff grows and jitter keeps waits in [0.5x, 1.5x) of the base
        assert 0.5 <= sleeps[0] < 1.5
        assert 1.0 <= sleeps[1] < 3.0

    def test_non_transient_raises_immediately(self):
        sleeps = []

        def bad():
            raise ValueError("invalid api key")

        with pytest.raises(ValueError):
            retry_call(bad, attempts=5, base_delay=1.0, sleep=sleeps.append)
        assert sleeps == []

    def test_exhausts_attempts_on_persistent_transient(self):
        sleeps = []

        def always_down():
            raise Exception("request timed out")

        with pytest.raises(Exception, match="timed out"):
            retry_call(always_down, attempts=3, base_delay=0.001, sleep=sleeps.append)
        # first try + 3 attempts -> 2 sleeps between them
        assert len(sleeps) == 2

    @pytest.mark.parametrize(
        "error,transient",
        [
            ("Rate limit reached", True),
            ("Error code: 429", True),
            ("Request timed out", True),
            ("Connection error", True),
            ("The server is overloaded", True),
            ("Incorrect API key", False),
            ("Invalid request body", False),
        ],
    )
    def test_error_classification(self, error, transient):
        assert is_transient_error(Exception(error)) is transient


# ---------------------------------------------------------------- cascade

class TestModelCascade:
    def test_falls_back_to_backup_model(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = wire_agent(agent, CascadeCompletions(
            fail_models=["primary-model"],
            content="from backup",
            usage=FakeUsage(10, 5),
        ))
        response = agent.call_llm([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "from backup"
        # primary retried (attempts=2), then the backup succeeded
        assert fake.models_used == ["primary-model", "primary-model", "backup-model"]
        # usage recorded from the successful call
        assert agent.get_usage()["prompt_tokens"] == 10
        assert agent.get_usage()["requests"] == 1

    def test_all_models_fail_raises(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = wire_agent(agent, CascadeCompletions(fail_models=["primary-model", "backup-model"]))
        with pytest.raises(Exception, match="LLM call failed"):
            agent.call_llm([{"role": "user", "content": "hi"}])
        assert fake.models_used == ["primary-model", "primary-model", "backup-model", "backup-model"]

    def test_non_transient_fails_over_without_retries(self, workdir):
        # A hard error on the primary (e.g. model not found) should not
        # burn retry attempts before trying the backup
        agent = OpenRouterAgent(silent=True)
        calls = {"n": 0}

        class HardFailCompletions:
            def create(self, **kwargs):
                calls["n"] += 1
                if kwargs["model"] == "primary-model":
                    raise Exception("model not found: primary-model")
                return FakeResponse(FakeMessage("backup answer"))

        wire_agent(agent, HardFailCompletions())
        response = agent.call_llm([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "backup answer"
        assert calls["n"] == 2  # one call per model, no retries

    def test_no_fallbacks_configured(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "k"
  model: "only-model"
system_prompt: "sys"
harness:
  retries:
    attempts: 1
    base_delay: 0.001
""",
            encoding="utf-8",
        )
        agent = OpenRouterAgent(silent=True)
        fake = wire_agent(agent, CascadeCompletions(fail_models=["only-model"]))
        with pytest.raises(Exception, match="LLM call failed"):
            agent.call_llm([{"role": "user", "content": "hi"}])
        assert fake.models_used == ["only-model"]


# ------------------------------------------------------------- reflection

class TestReflection:
    def make_harness(self, workdir, scripted):
        harness = AgentHarness(silent=True)
        fake = FakeCompletions_cls(scripted)
        harness.agent.client = type("C", (), {})()
        harness.agent.client.chat = type("Ch", (), {})()
        harness.agent.client.chat.completions = fake
        return harness, fake

    def test_critic_needs_fix_triggers_reanswer(self, workdir):
        enable_reflection(workdir)
        harness, fake = self.make_harness(workdir, [
            FakeMessage("The Eiffel Tower is in London."),
            FakeMessage("NEEDS_FIX: wrong city, it is in Paris."),
            FakeMessage("The Eiffel Tower is in Paris."),
        ])
        result = harness.query("where is the Eiffel Tower?")
        assert result == "The Eiffel Tower is in Paris."
        assert fake.calls == 3

    def test_critic_approved_returns_original(self, workdir):
        enable_reflection(workdir)
        harness, fake = self.make_harness(workdir, [
            FakeMessage("42"),
            FakeMessage("APPROVED"),
        ])
        result = harness.query("what is 6*7?")
        assert result == "42"
        assert fake.calls == 2

    def test_reflection_disabled_by_default(self, workdir):
        harness, fake = self.make_harness(workdir, [FakeMessage("answer")])
        result = harness.query("anything")
        assert result == "answer"
        assert fake.calls == 1


def FakeCompletions_cls(scripted):
    class _Fake:
        def __init__(self):
            self.scripted = list(scripted)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return FakeResponse(self.scripted.pop(0))

    return _Fake()


# -------------------------------------------------------------- transcript

class TestTranscript:
    def test_original_input_preserved_verbatim(self, workdir):
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, CascadeCompletions(
            fail_models=[], content="final answer", usage=FakeUsage(10, 5)
        ))

        # Long enough to trigger compression on the wire (threshold=100)
        long_input = "The launch budget is exactly 7 million dollars. " + "background filler. " * 40
        result = agent.run(long_input)
        assert result == "final answer"

        transcript_path = workdir / "logs" / "transcript.jsonl"
        assert transcript_path.exists()
        entries = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        # The ORIGINAL, uncompressed input - this is the no-data-loss guarantee
        assert entries[0]["input"] == long_input
        assert entries[0]["output"] == "final answer"
        assert entries[0]["usage"]["requests"] == 1

    def test_wire_copy_was_actually_compressed(self, workdir):
        agent = OpenRouterAgent(silent=True)
        fake = wire_agent(agent, CascadeCompletions(fail_models=[], content="ok"))

        long_input = "Key fact: the deadline is March 3rd. " + "filler sentence here. " * 30
        agent.run(long_input)
        sent = fake.last_messages[0][1]["content"]
        assert len(sent) < len(long_input)
        # And the original survived in the transcript anyway
        entry = json.loads((workdir / "logs" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["input"] == long_input

    def test_transcript_disabled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "k"
  model: "m"
system_prompt: "sys"
harness:
  transcript:
    enabled: false
""",
            encoding="utf-8",
        )
        agent = OpenRouterAgent(silent=True)
        wire_agent(agent, CascadeCompletions(fail_models=[], content="ok"))
        agent.run("hello")
        assert not (tmp_path / "logs" / "transcript.jsonl").exists()


# -----------------------------------------------------------------------
