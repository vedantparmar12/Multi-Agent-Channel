"""Tests for dynamic subagent spawning (triage) and plan-and-execute."""

import json

import pytest

import orchestrator
from orchestrator import TaskOrchestrator, build_fallback_questions
import planning as planning_module
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
    progress_excerpt_chars: 50
memory:
  enabled: true
  directory: memory
""",
        encoding="utf-8",
    )
    return tmp_path


class FakeAgent:
    """Scripted OpenRouterAgent stand-in.

    Instances consume responses from a shared FIFO queue; ``runs`` records
    every prompt so tests can assert on what agents actually received.
    """

    queue = []
    runs = []
    created = 0

    def __init__(self, silent=False, **kwargs):
        self.silent = silent
        self.tools = [{"type": "function", "function": {"name": "some_tool"}}]
        self.tool_mapping = {"some_tool": lambda **kw: {}}
        FakeAgent.created += 1

    def run(self, prompt, **kwargs):
        FakeAgent.runs.append(prompt)
        if not FakeAgent.queue:
            raise AssertionError(f"unexpected agent.run() call, no scripted response left: {prompt[:80]}")
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


TRIAGE_SIMPLE = json.dumps({"complexity": "simple", "num_agents": 1, "reasoning": "single fact"})
TRIAGE_COMPLEX = json.dumps({"complexity": "complex", "num_agents": 6, "reasoning": "broad survey"})
QUESTIONS_6 = json.dumps([f"question {i}" for i in range(6)])


# ------------------------------------------------------------ triage parse

class TestTriageParsing:
    def test_plain_json(self, workdir):
        orch = TaskOrchestrator(silent=True)
        parsed = orch._parse_json_object(TRIAGE_COMPLEX)
        assert parsed["complexity"] == "complex"
        assert parsed["num_agents"] == 6

    def test_markdown_fenced(self, workdir):
        orch = TaskOrchestrator(silent=True)
        assert orch._parse_json_object(f"```json\n{TRIAGE_SIMPLE}\n```")["complexity"] == "simple"

    def test_json_wrapped_in_prose(self, workdir):
        orch = TaskOrchestrator(silent=True)
        assert orch._parse_json_object(f"Sure! Here you go:\n{TRIAGE_COMPLEX}\nHope that helps.")["num_agents"] == 6

    def test_garbage_returns_none(self, workdir):
        orch = TaskOrchestrator(silent=True)
        assert orch._parse_json_object("no json here at all") is None
        assert orch._parse_json_object('["a list"]') is None

    @pytest.mark.parametrize("raw,count", [
        (json.dumps({"complexity": "complex", "num_agents": 99, "reasoning": ""}), 6),
        (json.dumps({"complexity": "complex", "num_agents": 0, "reasoning": ""}), 1),
        (json.dumps({"complexity": "complex", "num_agents": "three", "reasoning": ""}), 4),
    ])
    def test_num_agents_clamped_and_coerced(self, workdir, fake_agent, raw, count):
        FakeAgent.reset(raw)
        orch = TaskOrchestrator(silent=True)
        triage = orch.triage_request("anything")
        assert triage["num_agents"] == count

    def test_triage_failure_falls_back_to_config(self, workdir, fake_agent):
        FakeAgent.reset(Exception("API down"))
        orch = TaskOrchestrator(silent=True)
        triage = orch.triage_request("anything")
        assert triage["complexity"] == "complex"
        assert triage["num_agents"] == 4  # configured parallel_agents

    def test_invalid_complexity_falls_back(self, workdir, fake_agent):
        FakeAgent.reset(json.dumps({"complexity": "medium", "num_agents": 5, "reasoning": ""}))
        orch = TaskOrchestrator(silent=True)
        triage = orch.triage_request("anything")
        assert triage["num_agents"] == 4


# ------------------------------------------------------- dynamic orchestrate

class TestDynamicOrchestrate:
    def test_simple_request_single_agent(self, workdir, fake_agent):
        # Triage says simple: exactly ONE agent call, no questions, no synthesis
        FakeAgent.reset(TRIAGE_SIMPLE, "Tokyo time is 14:00")
        orch = TaskOrchestrator(silent=True)
        result = orch.orchestrate("what time is it in Tokyo?")
        assert result == "Tokyo time is 14:00"
        assert FakeAgent.created == 2  # triage agent + single worker
        assert len(FakeAgent.runs) == 2  # no decomposer, no synthesis calls
        # runs[0] is the triage prompt; the single worker gets the raw question
        assert FakeAgent.runs[1].startswith("what time is it in Tokyo?")
        assert orch.active_count == 1

    def test_complex_request_uses_proposed_count(self, workdir, fake_agent):
        # Triage proposes 6: decompose asks for 6, 6 workers, synthesis
        FakeAgent.reset(
            TRIAGE_COMPLEX,        # triage agent
            QUESTIONS_6,           # question generation agent
            *["worker response"] * 6,
            "final synthesis",     # synthesis agent
        )
        orch = TaskOrchestrator(silent=True)
        result = orch.orchestrate("survey the AI agent landscape")
        assert result == "final synthesis"
        assert orch.active_count == 6
        # decompose asked for exactly 6 questions
        assert "6" in FakeAgent.runs[1]
        # 6 worker prompts carry the generated questions
        worker_prompts = FakeAgent.runs[2:8]
        assert [p for p in worker_prompts if "question 0" in p]

    def test_triage_failure_degrades_to_configured_count(self, workdir, fake_agent):
        FakeAgent.reset(
            Exception("timeout"),  # triage fails
            QUESTIONS_6,           # decompose gets num_agents=4, but returns 6 questions...
            *["worker response"] * 4,  # ...which mismatch -> fallback questions used for 4
            "final synthesis",
        )
        orch = TaskOrchestrator(silent=True)
        result = orch.orchestrate("something complex")
        assert result == "final synthesis"
        assert orch.active_count == 4  # configured fallback count

    def test_dynamic_disabled_keeps_old_behavior(self, workdir, fake_agent):
        config_path = workdir / "config.yaml"
        config = config_path.read_text(encoding="utf-8").replace("dynamic: true", "dynamic: false")
        config_path.write_text(config, encoding="utf-8")

        FakeAgent.reset(
            QUESTIONS_6,
            *["worker response"] * 4,
            "final synthesis",
        )
        orch = TaskOrchestrator(silent=True)
        result = orch.orchestrate("anything")
        assert result == "final synthesis"
        assert FakeAgent.created == 1 + 4 + 1
        # no triage call happened
        assert FakeAgent.runs[0] == QUESTIONS_6 or "questions for" in FakeAgent.runs[0]

    def test_worker_failure_synthesizes_survivors(self, workdir, fake_agent):
        # 3 workers, one fails mid-flight: the 2 survivors still reach synthesis
        FakeAgent.reset(
            json.dumps({"complexity": "complex", "num_agents": 3, "reasoning": ""}),
            json.dumps(["q1", "q2", "q3"]),
            "good response",
            "better response",
            Exception("worker exploded"),
            "synthesis of survivors",
        )
        orch = TaskOrchestrator(silent=True)
        result = orch.orchestrate("three-sided comparison")
        assert result == "synthesis of survivors"


# ------------------------------------------------------------- plan parsing

class TestPlanParsing:
    def test_plain_json_steps(self):
        raw = json.dumps([{"step": 1, "description": "research"}, {"step": 2, "description": "write"}])
        steps = PlanExecutor._parse_steps(raw)
        assert [s["description"] for s in steps] == ["research", "write"]

    def test_markdown_fenced_and_prose(self):
        raw = f"Plan:\n```json\n{json.dumps([{'step': 1, 'description': 'only'}])}\n```"
        assert PlanExecutor._parse_steps(raw)[0]["description"] == "only"

    def test_missing_step_numbers_are_filled(self):
        raw = json.dumps([{"description": "a"}, {"description": "b"}])
        steps = PlanExecutor._parse_steps(raw)
        assert [s["step"] for s in steps] == [1, 2]

    def test_garbage_returns_none(self):
        assert PlanExecutor._parse_steps("I could not plan that") is None
        assert PlanExecutor._parse_steps('{"not": "a list"}') is None
        assert PlanExecutor._parse_steps("[]") is None


# ---------------------------------------------------------- plan execution

class TestPlanExecution:
    def test_fresh_agent_per_step_with_clean_context(self, workdir, fake_agent):
        # Planner + 3 step agents + synthesizer
        FakeAgent.reset(
            json.dumps([{"step": 1, "description": "gather sources"},
                        {"step": 2, "description": "analyze data"},
                        {"step": 3, "description": "write report"}]),
            "step 1 result: found 5 sources",
            "step 2 result: analysis done",
            "step 3 result: report written",
            "final assembled answer",
        )
        executor = PlanExecutor(silent=True)
        outcome = executor.execute("produce a market analysis")

        assert outcome["final"] == "final assembled answer"
        assert len(outcome["plan"]) == 3
        # One agent per step + planner + synthesizer: fresh context each
        assert FakeAgent.created == 5

        # Step 1 sees no prior progress
        step1_prompt = FakeAgent.runs[1]
        assert "starting - no steps completed yet" in step1_prompt
        # Step 2 sees step 1's excerpt, not step 1's full transcript
        step2_prompt = FakeAgent.runs[2]
        assert "found 5 sources" in step2_prompt
        # Step 3 sees step 2's excerpt
        step3_prompt = FakeAgent.runs[3]
        assert "analysis done" in step3_prompt
        # Every step prompt carries the overall task and its own assignment
        for prompt in FakeAgent.runs[1:4]:
            assert "produce a market analysis" in prompt
        assert "gather sources" in step1_prompt

    def test_progress_excerpt_is_bounded(self, workdir, fake_agent):
        FakeAgent.reset(
            json.dumps([{"step": 1, "description": "verbose step"},
                        {"step": 2, "description": "short step"}]),
            "x" * 2000,  # very verbose step 1
            "step 2 done",
            "synthesized",
        )
        executor = PlanExecutor(silent=True)
        executor.execute("task")
        step2_prompt = FakeAgent.runs[2]
        # progress_excerpt_chars=50 in the fixture config
        assert "x" * 60 not in step2_prompt
        assert step2_prompt.count("x") <= 200

    def test_planner_failure_becomes_single_step(self, workdir, fake_agent):
        FakeAgent.reset(
            Exception("planner API down"),
            "direct answer to the whole task",
        )
        executor = PlanExecutor(silent=True)
        outcome = executor.execute("small task")
        assert outcome["plan"] == [{"step": 1, "description": "small task"}]
        assert outcome["final"] == "direct answer to the whole task"
        # single result returns directly, no synthesis call
        assert FakeAgent.created == 2

    def test_plan_capped_at_max_steps(self, workdir, fake_agent):
        long_plan = json.dumps([{"step": i, "description": f"step {i}"} for i in range(1, 8)])
        FakeAgent.reset(long_plan)
        executor = PlanExecutor(silent=True)
        plan = executor.create_plan("big task")
        assert len(plan) == 3  # max_steps from fixture config

    def test_synthesis_failure_returns_step_results(self, workdir, fake_agent):
        FakeAgent.reset(
            json.dumps([{"step": 1, "description": "a"}, {"step": 2, "description": "b"}]),
            "result one",
            "result two",
            Exception("synthesis API down"),
        )
        executor = PlanExecutor(silent=True)
        outcome = executor.execute("task")
        assert "result one" in outcome["final"]
        assert "result two" in outcome["final"]


# --------------------------------------------------------- harness :plan

class TestHarnessPlanCommand:
    def test_plan_execute_routes_to_executor(self, workdir, fake_agent, monkeypatch):
        from harness import AgentHarness

        FakeAgent.reset(
            json.dumps([{"step": 1, "description": "do the thing"}]),
            "step result",
        )
        harness = AgentHarness(silent=True)
        result = harness.plan_execute("build a thing")
        assert result == "step result"  # single step returns directly
        assert FakeAgent.created >= 2  # harness's own agent + planner + step agent

        # The plan landed in the memory log
        log = (workdir / "memory" / "log.md").read_text(encoding="utf-8")
        assert "plan | build a thing" in log
