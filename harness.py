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

        self.agent = OpenRouterAgent(config_path=config_path, silent=silent)
        self.memory_store = self.agent.memory_store

        # Per-run report, filled in by query()/ingest()/lint()
        self.last_report = {}

    def query(self, user_input: str) -> str:
        """Answer a question with the full agent loop (tools + memory)."""
        return self._execute("query", user_input)

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
        # Snapshot usage so the report covers only this run
        before = self.agent.get_usage()
        response = self.agent.run(prompt)
        after = self.agent.get_usage()

        # Bookkeeping: every operation lands in the parseable memory log
        if self.memory_store is not None:
            try:
                title = log_title or (prompt.strip().splitlines()[0][:80])
                self.memory_store.append_log("query" if action == "query" else action, title)
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
    print("Commands: :ingest <paste source>  |  :lint  |  anything else is a query")
    print("-" * 60)

    try:
        harness = AgentHarness()
        print(f"Using model: {harness.config['openrouter']['model']}")
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
