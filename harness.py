"""Agent harness: the code around the loop that makes it an agent.

An agent is three things working together (the Karpathy formulation):
a while-loop over an LLM, a set of tools, and notes the agent itself
maintains. ``OpenRouterAgent`` already provides the loop and the tools;
this module adds the third pillar and the glue:

- **Memory** - the persistent markdown wiki (``memory.py``) is injected
  into every system prompt, and the agent files knowledge itself through
  the ``save_memory_page`` tool. Knowledge compounds across runs instead
  of vanishing into chat history.
- **Token proxy** - ``token_budget.py`` sits between the loop and the
  API, shrinking every request (input compression, tool-result
  truncation, history compaction) so loop iterations don't compound cost.
- **Operations** - the three wiki operations: query (answer + auto-log),
  ingest (file a source into the wiki), and lint (health-check the wiki).

Run interactively:

    python harness.py

Or programmatically:

    from harness import AgentHarness
    harness = AgentHarness()
    answer = harness.query("what did we decide about caching?")
"""

import yaml
from agent import OpenRouterAgent


class AgentHarness:
    """Shell around OpenRouterAgent adding memory, budget, and reporting."""

    def __init__(self, config_path="config.yaml", silent=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.silent = silent
        self.config_path = config_path

        self.agent = OpenRouterAgent(config_path=config_path, silent=silent)
        self.memory_store = self.agent.memory_store

        # Reflection (quality mode): a critic pass reviews each answer and
        # re-answers the request when it finds concrete flaws. Doubles token
        # cost per query in the worst case, so it is opt-in.
        self.reflection_enabled = (
            self.config.get('harness', {}).get('reflection', {}).get('enabled', False)
        )

        # Per-run report, filled in by query()/ingest()/lint()
        self.last_report = {}

    def query(self, user_input: str) -> str:
        """Answer a question with the full agent loop (tools + memory)."""
        before = self.agent.get_usage()
        answer = self.agent.run(user_input)
        if self.reflection_enabled and answer:
            answer = self._reflect(user_input, answer)
        return self._finish("query", user_input, before, answer)

    def plan_execute(self, task: str) -> str:
        """Run a big task plan-and-execute: planner emits JSON steps, each
        step runs in a fresh subagent with a clean context window, and a
        final synthesis assembles the answer."""
        from planning import PlanExecutor

        before = self.agent.get_usage()
        outcome = PlanExecutor(config_path=self.config_path, silent=self.silent).execute(task)
        # Report the harness agent's usage only; step agents are separate
        # instances (that is the point - clean windows), so their usage is
        # reported by their own runs
        answer = outcome["final"]
        self._finish("plan", task, before, answer)
        if not self.silent:
            print(f"🧩 Plan: {len(outcome['plan'])} step(s) executed")
        return answer

    def _reflect(self, question: str, answer: str) -> str:
        """Adversarial critic pass: verify the answer, re-answer if flawed.

        The critic runs through the same agent loop, so it can use tools
        (e.g. web search) to fact-check claims. One reflection round max -
        endless self-critique loops are how agents stall.
        """
        critic_prompt = (
            "Review this answer for factual errors, unsupported claims, or "
            "missed parts of the request. Use tools to verify claims when "
            "useful.\n\n"
            f"REQUEST:\n{question}\n\nANSWER:\n{answer}\n\n"
            "Reply exactly 'APPROVED' if the answer is good. Otherwise reply "
            "'NEEDS_FIX:' followed by the specific problems."
        )
        critique = self.agent.run(critic_prompt)
        if "NEEDS_FIX" not in critique.upper():
            return answer
        retry_prompt = (
            f"{question}\n\nA reviewer found problems with a previous "
            f"attempt:\n{critique}\n\nAddress every problem and answer again."
        )
        return self.agent.run(retry_prompt)

    def ingest(self, source_text: str) -> str:
        """File a raw source into the wiki: the agent summarizes it, updates
        entity/concept pages, and refreshes the index."""
        prompt = (
            "Ingest the following source into your persistent memory wiki.\n"
            "1. Read it and decide what is durable knowledge.\n"
            "2. Save or update the relevant wiki pages with save_memory_page "
            "(update existing pages instead of duplicating them).\n"
            "3. Cross-reference related pages you already have.\n"
            "4. End with a one-paragraph summary of what you filed.\n\n"
            "--- SOURCE START ---\n"
            f"{source_text}\n"
            "--- SOURCE END ---"
        )
        return self._execute("ingest", prompt, log_title="source ingest")

    def lint(self) -> str:
        """Health-check the wiki: contradictions, stale claims, orphan pages."""
        prompt = (
            "Lint your persistent memory wiki. Review the pages listed in your "
            "memory index and report:\n"
            "- contradictions between pages\n"
            "- stale or superseded claims\n"
            "- important concepts mentioned in pages but lacking their own page\n"
            "- missing cross-references\n"
            "Fix what you can directly with save_memory_page, then summarize "
            "what you changed."
        )
        return self._execute("lint", prompt, log_title="wiki lint pass")

    def _execute(self, action: str, prompt: str, log_title: str = None) -> str:
        before = self.agent.get_usage()
        response = self.agent.run(prompt)
        return self._finish(action, log_title or prompt, before, response)

    def _finish(self, action: str, title: str, before: dict, response: str) -> str:
        """Bookkeeping shared by every operation: memory log + usage report."""
        after = self.agent.get_usage()

        # Every operation lands in the parseable memory log
        if self.memory_store is not None:
            try:
                first_line = (title or "").strip().splitlines()[0][:80] if (title or "").strip() else "untitled"
                self.memory_store.append_log("query" if action == "query" else action, first_line)
            except Exception:
                pass

        budget = self.agent.budget.last_report
        self.last_report = {
            "action": action,
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "requests": after["requests"] - before["requests"],
            "tool_results_truncated": budget.get("tool_results_truncated", 0),
            "history_compacted": budget.get("compacted", False),
        }
        if not self.silent:
            self._print_report()
        return response

    def _print_report(self) -> None:
        report = self.last_report
        print(
            f"📊 tokens: {report['prompt_tokens']} in / "
            f"{report['completion_tokens']} out "
            f"({report['requests']} request(s))"
        )
        notes = []
        if report["tool_results_truncated"]:
            notes.append(f"{report['tool_results_truncated']} tool result(s) truncated")
        if report["history_compacted"]:
            notes.append("history compacted to fit budget")
        if notes:
            print(f"💰 cost guard: {', '.join(notes)}")


def main():
    """Interactive CLI for the agent harness."""
    print("Agent Harness - loop + tools + persistent memory")
    print("Type 'quit', 'exit', or 'bye' to exit")
    print("Commands: :ingest <source> | :lint | :plan <big task> | anything else is a query")
    print("-" * 60)

    try:
        harness = AgentHarness()
        print(f"Using model: {harness.config['openrouter']['model']}")
        if harness.agent.fallback_models:
            print(f"Fallbacks: {', '.join(harness.agent.fallback_models)}")
        if harness.reflection_enabled:
            print("Reflection: on (each answer gets a critic pass)")
        if harness.memory_store is not None:
            pages = harness.memory_store.list_pages()
            print(f"Memory: {len(pages)} page(s) loaded from the wiki")
        print("-" * 60)
    except Exception as e:
        print(f"Error initializing harness: {e}")
        print("1. Set your OpenRouter API key in config.yaml")
        print("2. Install dependencies with: pip install -r requirements.txt")
        return

    while True:
        try:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a question or command.")
                continue

            if user_input.lower() == ":lint":
                print("Agent: linting the memory wiki...")
                response = harness.lint()
            elif user_input.lower().startswith(":plan"):
                task = user_input[len(":plan"):].strip()
                if not task:
                    print("Usage: :plan <big task - planned into steps, each run by a fresh subagent>")
                    continue
                print("Agent: planning and executing...")
                response = harness.plan_execute(task)
            elif user_input.lower().startswith(":ingest"):
                source = user_input[len(":ingest"):].strip()
                if not source:
                    print("Usage: :ingest <text to file into the wiki>")
                    continue
                print("Agent: ingesting source into the wiki...")
                response = harness.ingest(source)
            else:
                print("Agent: thinking...")
                response = harness.query(user_input)

            print(f"Agent: {response}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
