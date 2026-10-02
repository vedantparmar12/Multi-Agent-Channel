"""Tests for the agent harness features: token budget proxy, Karpathy-style
memory wiki, and their wiring into the agent loop."""

import json

import pytest

from token_budget import TokenBudget, compress_text, estimate_tokens, estimate_messages_tokens
from memory import MemoryStore, slugify, LOG_ENTRY_RE
from tools.memory_tool import SaveMemoryPageTool, ReadMemoryPageTool
from harness import AgentHarness


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated directory with a minimal config.yaml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "test-model"
system_prompt: "You are a test agent."
agent:
  max_iterations: 5
harness:
  history_budget: 500
  tool_result_limit: 100
  input_compression_threshold: 500
  input_compression_ratio: 0.5
memory:
  enabled: true
  directory: memory
""",
        encoding="utf-8",
    )
    return tmp_path


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


class FakeCompletions:
    def __init__(self, scripted, usages=None):
        self.scripted = list(scripted)
        self.usages = list(usages or [])
        self.calls = 0
        self.sent_messages = []

    def create(self, **kwargs):
        self.calls += 1
        self.sent_messages.append(kwargs["messages"])
        usage = self.usages.pop(0) if self.usages else None
        return FakeResponse(self.scripted.pop(0), usage)


def make_tool_call(name, args):
    """Minimal duck-typed tool_call object."""
    class Function:
        pass

    class ToolCall:
        pass

    call = ToolCall()
    call.id = f"call_{name}"
    call.function = Function()
    call.function.name = name
    call.function.arguments = json.dumps(args)
    return call


# ------------------------------------------------------------ compression

class TestCompressText:
    def test_noop_below_target(self):
        text = "Short text. Nothing to do."
        assert compress_text(text, 1000) == text

    def test_reduces_size_and_keeps_key_facts(self):
        filler = ("This is some preamble filler material that goes on. " * 10)
        text = (
            filler
            + "The API costs 3 dollars per 1000 tokens. "
            + "Gpt-4o supports function calling. "
            + filler
        )
        compressed = compress_text(text, 200)
        assert len(compressed) < len(text)
        assert "3 dollars" in compressed
        assert "Gpt-4o" in compressed

    def test_first_sentence_always_kept(self):
        text = "What is the latency budget? " + "Filler sentence here. " * 30
        compressed = compress_text(text, 100)
        assert compressed.startswith("What is the latency budget?")


class TestTokenBudget:
    def test_truncates_tool_results(self):
        budget = TokenBudget({"tool_result_limit": 50, "history_budget": 10**9})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "1", "name": "search_web",
             "content": "x" * 500},
        ]
        prepared = budget.prepare_messages(messages)
        assert len(prepared[2]["content"]) < 100
        assert "truncated" in prepared[2]["content"]
        assert budget.last_report["tool_results_truncated"] == 1

    def test_short_tool_results_untouched(self):
        budget = TokenBudget({"tool_result_limit": 100})
        messages = [{"role": "tool", "tool_call_id": "1", "name": "t", "content": "small"}]
        assert budget.prepare_messages(messages)[0]["content"] == "small"

    def test_compacts_history_over_budget(self):
        # history_budget in est tokens: 1 token = 4 chars
        budget = TokenBudget({"history_budget": 100, "recent_turns_kept": 2})
        messages = [{"role": "system", "content": "system prompt"}]
        messages.append({"role": "user", "content": "original request"})
        for i in range(10):
            messages.append({"role": "assistant", "content": f"turn {i} " + "y" * 200})
        prepared = budget.prepare_messages(messages)
        assert budget.last_report["compacted"] is True
        # System prompt and original request survive verbatim
        assert prepared[0]["content"] == "system prompt"
        assert prepared[1]["content"] == "original request"
        # Middle turns collapsed into a digest message
        assert any("auto-compacted" in str(m.get("content", "")) for m in prepared)
        # Recent turns survive
        assert prepared[-1]["content"].startswith("turn 9")
        # And the result actually fits the budget
        assert estimate_messages_tokens(prepared) <= 100 + estimate_tokens("system prompt") + 200

    def test_under_budget_is_untouched(self):
        budget = TokenBudget({"history_budget": 10**9})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert budget.prepare_messages(messages) == messages

    def test_compress_user_input_threshold(self):
        budget = TokenBudget({"input_compression_threshold": 100, "input_compression_ratio": 0.5})
        short = "just a question"
        assert budget.compress_user_input(short) == short
        long_input = "Important fact: revenue was 42 million. " + "filler. " * 60
        compressed = budget.compress_user_input(long_input)
        assert len(compressed) < len(long_input)
        assert "42 million" in compressed


# ----------------------------------------------------------------- memory

class TestMemoryStore:
    def test_bootstrap_creates_structure(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        assert (tmp_path / "memory" / "wiki").is_dir()
        assert (tmp_path / "memory" / "index.md").exists()
        assert (tmp_path / "memory" / "log.md").exists()

    def test_save_page_updates_index_and_log(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        store.save_page("User Preferences", "Prefers concise answers.")

        assert "User Preferences" in store.get_index()
        page = store.read_page("User Preferences")
        assert "concise answers" in page

        log = (tmp_path / "memory" / "log.md").read_text(encoding="utf-8")
        matches = LOG_ENTRY_RE.findall(log)
        assert matches == [(matches[0][0], "save", "User Preferences")]
        # Log entries are grep-able with a stable prefix
        assert log.rstrip().splitlines()[-1].startswith("## [")

    def test_resave_updates_index_in_place(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        store.save_page("Caching", "v1 decision")
        store.save_page("Caching", "v2 decision")
        index = store.get_index()
        assert index.count("](wiki/caching.md)") == 1
        assert "v2 decision" in store.read_page("Caching")

    def test_read_missing_page_returns_none(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        assert store.read_page("no such page") is None

    def test_log_tail_ordered(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        store.append_log("query", "first")
        store.append_log("query", "second")
        store.append_log("save", "third")
        tail = store.get_log_tail(entries=2)
        assert len(tail) == 2
        # newest last
        assert "first" not in tail
        assert "third" in tail[-1]

    def test_prompt_section_contains_index_and_log(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        store.save_page("Rest API", "Use versioned endpoints.")
        section = store.get_prompt_section()
        assert "Agent Memory" in section
        assert "Rest API" in section
        assert "Known pages" in section

    def test_prompt_section_capped(self, tmp_path):
        store = MemoryStore(directory="memory", root=tmp_path)
        for i in range(50):
            store.save_page(f"Page {i}", "content " * 100)
        section = store.get_prompt_section(max_chars=800)
        assert len(section) <= 800 + len("\n[... memory index truncated ...]")
        assert "truncated" in section

    def test_slugify_blocks_path_traversal(self):
        assert slugify("../../etc/passwd") == "etc-passwd"
        assert slugify("a/b\\c") == "a-b-c"
        assert slugify("  ") == "untitled"


class TestMemoryTools:
    def test_save_and_read_roundtrip(self, workdir):
        save_tool = SaveMemoryPageTool({})
        result = save_tool.execute(title="Deployment", content="Use blue-green deploys.")
        assert result["status"] == "success"

        read_tool = ReadMemoryPageTool({})
        result = read_tool.execute(title="Deployment")
        assert result["status"] == "success"
        assert "blue-green" in result["content"]

    def test_read_missing_lists_available(self, workdir):
        SaveMemoryPageTool({}).execute(title="Known", content="something")
        result = ReadMemoryPageTool({}).execute(title="Unknown")
        assert "error" in result
        assert "Known" in result["available_pages"]

    def test_rejects_empty(self, workdir):
        result = SaveMemoryPageTool({}).execute(title="", content="")
        assert "error" in result

    def test_share_one_wiki(self, workdir):
        # The tool store and the agent store point at the same directory
        SaveMemoryPageTool({}).execute(title="Shared", content="visible to both")
        store = MemoryStore(directory="memory")
        assert "visible to both" in store.read_page("Shared")


# ------------------------------------------------------- agent integration

class TestAgentWiring:
    def make_agent(self, workdir):
        from agent import OpenRouterAgent
        agent = OpenRouterAgent(silent=True)
        # Avoid real network calls in tests
        agent.client = None
        return agent

    def test_memory_injected_into_system_prompt(self, workdir):
        from tools.memory_tool import SaveMemoryPageTool
        SaveMemoryPageTool({}).execute(title="Known Fact", content="the sky is blue")
        agent = self.make_agent(workdir)
        prompt = agent._build_system_prompt()
        assert "Agent Memory" in prompt
        assert "Known Fact" in prompt
        # Memory tools are discovered like every other tool
        assert "save_memory_page" in agent.tool_mapping
        assert "read_memory_page" in agent.tool_mapping

    def test_usage_accumulated_from_responses(self, workdir):
        agent = self.make_agent(workdir)
        fake = FakeCompletions(
            [FakeMessage("answer")],
            usages=[FakeUsage(100, 20)],
        )
        agent.client = type("C", (), {})()
        agent.client.chat = type("Ch", (), {})()
        agent.client.chat.completions = fake
        agent.call_llm([{"role": "user", "content": "hi"}])
        assert agent.get_usage() == {"prompt_tokens": 100, "completion_tokens": 20, "requests": 1}

    def test_wire_messages_are_prepared(self, workdir):
        agent = self.make_agent(workdir)
        long_tool_result = "z" * 2000  # over tool_result_limit=100 in fixture config
        fake = FakeCompletions([FakeMessage("done")])
        agent.client = type("C", (), {})()
        agent.client.chat = type("Ch", (), {})()
        agent.client.chat.completions = fake
        agent.call_llm([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "1", "name": "search_web",
             "content": long_tool_result},
        ])
        sent_tool_content = fake.sent_messages[0][2]["content"]
        assert len(sent_tool_content) < 200
        assert "truncated" in sent_tool_content


class TestAgentLoopWithMemory:
    def test_agent_files_knowledge_via_tool(self, workdir):
        from agent import OpenRouterAgent
        agent = OpenRouterAgent(silent=True)

        call = make_tool_call("save_memory_page", {
            "title": "User Preference",
            "content": "User wants terse responses.",
        })
        fake = FakeCompletions([
            FakeMessage("Let me save that.", tool_calls=[call]),
            FakeMessage("Saved your preference."),
        ])
        agent.client = type("C", (), {})()
        agent.client.chat = type("Ch", (), {})()
        agent.client.chat.completions = fake

        result = agent.run("remember I like terse responses")
        assert fake.calls == 2
        assert "Saved your preference" in result

        # The memory actually landed in the wiki
        store = MemoryStore(directory="memory")
        assert "terse responses" in store.read_page("User Preference")
        assert "User Preference" in store.get_index()

    def test_long_input_compressed_before_send(self, workdir):
        from agent import OpenRouterAgent
        agent = OpenRouterAgent(silent=True)
        fake = FakeCompletions([FakeMessage("ok")])
        agent.client = type("C", (), {})()
        agent.client.chat = type("Ch", (), {})()
        agent.client.chat.completions = fake

        long_input = "The budget is 7 million dollars for 2027. " + "filler text here. " * 80
        agent.run(long_input)
        sent = fake.sent_messages[0][1]["content"]
        assert len(sent) < len(long_input)
        assert "7 million" in sent


class TestHarness:
    def make_harness(self, workdir, scripted, usages=None):
        harness = AgentHarness(silent=True)
        fake = FakeCompletions(scripted, usages=usages)
        harness.agent.client = type("C", (), {})()
        harness.agent.client.chat = type("Ch", (), {})()
        harness.agent.client.chat.completions = fake
        return harness, fake

    def test_query_returns_answer_and_logs(self, workdir):
        harness, fake = self.make_harness(workdir, [FakeMessage("42")], usages=[FakeUsage(50, 10)])
        result = harness.query("what is the answer")
        assert result == "42"
        assert harness.last_report["prompt_tokens"] == 50
        assert harness.last_report["completion_tokens"] == 10
        assert harness.last_report["requests"] == 1

        log = (workdir / "memory" / "log.md").read_text(encoding="utf-8")
        assert "query | what is the answer" in log

    def test_ingest_prompt_wraps_source(self, workdir):
        harness, fake = self.make_harness(workdir, [FakeMessage("filed it")])
        harness.ingest("Some article text about caching strategies.")
        sent = fake.sent_messages[0][1]["content"]
        assert "SOURCE START" in sent
        assert "Some article text" in sent
        assert "save_memory_page" in sent

    def test_lint_prompt_checks_wiki(self, workdir):
        harness, fake = self.make_harness(workdir, [FakeMessage("wiki is healthy")])
        harness.lint()
        sent = fake.sent_messages[0][1]["content"]
        assert "contradictions" in sent
        log = (workdir / "memory" / "log.md").read_text(encoding="utf-8")
        assert "lint" in log

    def test_memory_persists_across_harness_runs(self, workdir):
        # Run 1: agent saves knowledge
        call = make_tool_call("save_memory_page", {
            "title": "Decision",
            "content": "Use Postgres not MySQL.",
        })
        harness, _ = self.make_harness(workdir, [
            FakeMessage("saving", tool_calls=[call]),
            FakeMessage("noted"),
        ])
        harness.query("remember: we chose Postgres")

        # Run 2: a fresh harness sees the knowledge in its system prompt
        harness2, fake2 = self.make_harness(workdir, [FakeMessage("we chose Postgres")])
        harness2.query("what database did we choose")
        system_prompt = fake2.sent_messages[0][0]["content"]
        assert "Decision" in system_prompt
        assert "Postgres" in system_prompt
