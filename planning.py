"""Plan-and-execute: multi-step tasks as sequential subagents.

For tasks too big for one loop ("build X", "research A then design B"),
a planner turns the request into a short list of steps, and each step
runs in its own agent with a clean context window - the only context a
step sees is the task, a compact summary of what previous steps produced,
and its own instruction. Long tasks no longer drown in a growing
transcript; each step starts fresh.

Steps run sequentially because each depends on the previous results;
parallel fan-out is the orchestrator's job (orchestrator.py).

Fallbacks everywhere: a failed planner becomes a single-step plan, a
failed synthesizer returns the concatenated step results.
"""

import json
import re
from typing import Dict, List, Optional

import yaml

from agent import OpenRouterAgent

DEFAULT_MAX_STEPS = 10

DEFAULT_PLANNER_PROMPT = """You are a task planner. Break this task into the
minimum number of sequential steps that one capable AI agent each can
complete independently. Fewer, meatier steps beat many tiny ones - most
tasks need 2-5.

Task: "{task}"

Each step must be self-contained: the executing agent sees the task, a
summary of earlier step results, and the step text - nothing else.

Respond with ONLY a JSON array and nothing else:
[{{"step": 1, "description": "..."}}, {{"step": 2, "description": "..."}}]"""

DEFAULT_STEP_PROMPT = """Overall task: {task}

Progress so far:
{progress}

Your assignment (step {step_number} of {total_steps}):
{description}

Complete only this step. Return the step's result, nothing else."""

DEFAULT_SYNTHESIS_PROMPT = """You are a synthesis agent. The overall task was:

"{task}"

It was executed in steps. The results follow:

{results}

Synthesize the results into one coherent, complete final answer to the
original task. Resolve contradictions between steps, keep the strongest
details, and respond directly with the final answer only."""


class PlanExecutor:
    """Plans a task into steps and executes them as sequential subagents."""

    def __init__(self, config_path="config.yaml", silent=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.silent = silent

        planning_config = self.config.get("harness", {}).get("planning", {})
        self.max_steps = int(planning_config.get("max_steps", DEFAULT_MAX_STEPS))
        self.planner_prompt_template = planning_config.get("planner_prompt", DEFAULT_PLANNER_PROMPT)
        # Each step result is summarized to roughly this many characters when
        # carried forward, so step N's prompt stays bounded regardless of
        # how verbose step N-1 was
        self.progress_excerpt_chars = int(planning_config.get("progress_excerpt_chars", 600))

    # -- planning ----------------------------------------------------------

    def create_plan(self, task: str) -> List[Dict[str, str]]:
        """Turn a task into a capped list of steps.

        Returns [{"step": 1, "description": "..."}, ...]. Falls back to a
        single-step plan (the task itself) whenever the planner fails or
        returns nonsense - execution never stops at the planner.
        """
        try:
            planner = OpenRouterAgent(silent=True)
            planner.tools = []
            planner.tool_mapping = {}
            prompt = self.planner_prompt_template.format(task=task)
            steps = self._parse_steps(planner.run(prompt))
        except Exception as e:
            if not self.silent:
                print(f"⚠️  Planner failed ({e}); running as a single step")
            steps = None

        if not steps:
            return [{"step": 1, "description": task}]

        # Hard cap: a runaway planner cannot produce a 50-step bill
        if len(steps) > self.max_steps:
            if not self.silent:
                print(f"⚠️  Plan had {len(steps)} steps; capping at {self.max_steps}")
            steps = steps[: self.max_steps]
        return steps

    @staticmethod
    def _parse_steps(response: str) -> Optional[List[Dict[str, str]]]:
        """Parse a JSON step array, tolerating prose and markdown fences."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

        candidates = [text]
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            candidates.append(bracket_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list) or not parsed:
                continue
            steps = []
            for i, item in enumerate(parsed, 1):
                if not isinstance(item, dict):
                    continue
                description = str(item.get("description", "")).strip()
                if not description:
                    continue
                try:
                    number = int(item.get("step", i))
                except (TypeError, ValueError):
                    number = i
                steps.append({"step": number, "description": description})
            if steps:
                return steps
        return None

    # -- execution ---------------------------------------------------------

    def execute(self, task: str) -> Dict[str, object]:
        """Plan and run a task. Returns plan, per-step results, final answer."""
        plan = self.create_plan(task)
        total = len(plan)

        results: List[str] = []
        progress_lines: List[str] = []
        for index, step in enumerate(plan, 1):
            if not self.silent:
                print(f"▶ Step {index}/{total}: {step['description'][:80]}")

            # Fresh agent per step: clean context window, full tool access.
            # The prompt carries only the task, compact prior progress, and
            # this step's instruction - the step never sees prior transcripts.
            step_agent = OpenRouterAgent(silent=True)
            prompt = DEFAULT_STEP_PROMPT.format(
                task=task,
                progress="\n".join(progress_lines) if progress_lines else "(starting - no steps completed yet)",
                step_number=index,
                total_steps=total,
                description=step["description"],
            )
            results.append(step_agent.run(prompt))
            progress_lines.append(f"Step {index} ({step['description'][:60]}): {results[-1][:self.progress_excerpt_chars]}")

        final = self._synthesize(task, results)
        return {"plan": plan, "results": results, "final": final}

    def _synthesize(self, task: str, results: List[str]) -> str:
        """Combine step results into the final answer for the task."""
        if len(results) == 1:
            return results[0]
        try:
            synthesizer = OpenRouterAgent(silent=True)
            synthesizer.tools = []
            synthesizer.tool_mapping = {}
            results_text = "\n\n".join(
                f"=== STEP {i} RESULT ===\n{result}" for i, result in enumerate(results, 1)
            )
            prompt = DEFAULT_SYNTHESIS_PROMPT.format(task=task, results=results_text)
            return synthesizer.run(prompt)
        except Exception:
            # A failed synthesis must not lose the step results
            return "\n\n".join(
                f"=== Step {i} ===\n{result}" for i, result in enumerate(results, 1)
            )
