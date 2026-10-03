import json
import sys
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from tools import discover_tools
from token_budget import TokenBudget, _tool_call_arguments, _tool_call_name
from reliability import retry_call

# Windows consoles often default to cp1252, where the progress emoji in the
# non-silent prints below crash with UnicodeEncodeError (and the error
# handler prints an emoji too, so it crashes twice). Degrade to replacement
# characters instead of dying. agent.py is imported by every entry point,
# so this covers main.py, orchestrator.py, and harness.py as well.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def structured_output(name: str, schema: dict) -> dict:
    """response_format payload for provider-native structured outputs.

    Callers pass this to ``agent.run(prompt, response_format=...)`` so the
    model is constrained to emit schema-valid JSON instead of relying on
    prompt instructions and lenient parsing.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


# Error texts that mean "this provider does not support response_format";
# those are worth one plain retry, unlike ordinary outages
STRUCTURED_OUTPUT_HINTS = (
    "response_format", "json_schema", "json mode", "json_object",
    "structured output",
)

class OpenRouterAgent:
    def __init__(self, config_path="config.yaml", silent=False, context_aware=True):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Silent mode for orchestrator (suppresses debug output)
        self.silent = silent
        self.context_aware = context_aware

        # Token budget middleware: every outgoing request passes through
        # prepare_messages, which truncates tool results and compacts old
        # history so loop iterations don't compound input cost
        self.budget = TokenBudget(self.config.get('harness', {}))

        # Reliability: retries with backoff for transient API errors, and a
        # fallback model cascade for when the primary model keeps failing
        retry_config = self.config.get('harness', {}).get('retries', {})
        self.retry_attempts = int(retry_config.get('attempts', 3))
        self.retry_base_delay = float(retry_config.get('base_delay', 1.0))
        self.fallback_models = list(self.config['openrouter'].get('fallback_models') or [])

        # Local transcript: every original prompt and answer is appended to
        # logs/transcript.jsonl. The wire copy of an input may be compressed,
        # but the original always stays on disk - token savings never cost
        # data.
        transcript_config = self.config.get('harness', {}).get('transcript', {})
        self.transcript_enabled = transcript_config.get('enabled', True)
        self.transcript_path = Path('logs') / 'transcript.jsonl'

        # Usage accumulated across all API calls in this agent's lifetime.
        # cached_tokens and cost come from OpenRouter's usage details, so
        # spend reporting (:cost in harness.py) reflects real money.
        self.usage_totals = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "cost": 0.0, "requests": 0,
        }
        self.usage_by_model = {}

        # OpenRouter session id: pins requests to one provider endpoint so
        # prompt-cache prefixes actually hit across the loop's repeated
        # calls. Auto-generated per agent unless configured.
        self.session_id = self.config['openrouter'].get('session_id') or uuid.uuid4().hex[:16]

        # Persistent markdown-wiki memory (Karpathy-style). Memory tools in
        # tools/memory_tool.py build their own store pointed at the same
        # directory, so the agent reads and writes one wiki.
        self.memory_store = None
        if self.config.get('memory', {}).get('enabled', True):
            try:
                from memory import MemoryStore
                self.memory_store = MemoryStore(
                    directory=self.config['memory'].get('directory', 'memory')
                )
            except Exception as e:
                if not self.silent:
                    print(f"⚠️  Memory disabled: {e}")

        # Initialize context loader if context awareness is enabled
        self.context_loader = None
        self.project_context = None
        if self.context_aware:
            try:
                from context import ContextLoader
                self.context_loader = ContextLoader()
                self.project_context = self.context_loader.load_project_context()
                if not self.silent:
                    print("✅ Context loaded successfully")
            except Exception as e:
                if not self.silent:
                    print(f"⚠️  Context loading failed: {e}")
                self.context_aware = False
        
        # Initialize OpenAI client with OpenRouter
        self.client = OpenAI(
            base_url=self.config['openrouter']['base_url'],
            api_key=self.config['openrouter']['api_key'],
            # Bounded requests keep hung API calls from blocking the agent
            # loop and the orchestrator's thread pool forever
            timeout=self.config['openrouter'].get('request_timeout', 120)
        )
        
        # Discover tools dynamically
        self.discovered_tools = discover_tools(self.config, silent=self.silent)
        
        # Build OpenRouter tools array
        self.tools = [tool.to_openrouter_schema() for tool in self.discovered_tools.values()]
        
        # Build tool mapping
        self.tool_mapping = {name: tool.execute for name, tool in self.discovered_tools.items()}
    
    
    def call_llm(self, messages, response_format: dict = None):
        """Make an OpenRouter API call with retries and model fallback.

        The token-budget proxy runs first (messages are truncated/compacted
        just before hitting the wire), then each model is attempted with
        backoff on transient errors; non-transient errors fail over to the
        next model immediately since retrying cannot fix them.

        When ``response_format`` is set and the provider rejects structured
        outputs, the call degrades once to plain text instead of failing -
        callers parse leniently, so the run continues either way.
        """
        try:
            return self._call_llm_format(messages, response_format)
        except Exception as e:
            if response_format is None:
                raise
            message = str(e).lower()
            if not any(hint in message for hint in STRUCTURED_OUTPUT_HINTS):
                raise
            if not self.silent:
                print("⚠️  structured outputs unavailable on this provider; retrying as plain text")
            return self._call_llm_format(messages, None)

    def _call_llm_format(self, messages, response_format: dict = None):
        prepared = self.budget.prepare_messages(messages)
        models = [self.config['openrouter']['model']] + self.fallback_models
        last_error = None

        for model in models:
            try:
                response = retry_call(
                    lambda m=model, r=response_format: self._create_completion(m, prepared, r),
                    attempts=self.retry_attempts,
                    base_delay=self.retry_base_delay,
                )
                self._record_usage(response, model)
                return response
            except Exception as e:
                last_error = e
                if not self.silent:
                    print(f"⚠️  model '{model}' unavailable: {e}")

        raise Exception(f"LLM call failed: {last_error}")

    def _create_completion(self, model, messages, response_format: dict = None):
        request_kwargs = {
            "model": model,
            "messages": messages,
            # OpenRouter routing hint: keeps the loop's requests on the
            # same provider endpoint so prompt-cache prefixes survive
            "extra_body": {"session_id": self.session_id},
        }
        # An empty tools array is rejected by many providers
        if self.tools:
            request_kwargs["tools"] = self.tools
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        return self.client.chat.completions.create(**request_kwargs)

    def _record_usage(self, response, model: str):
        """Accumulate real usage numbers returned by the API."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        # prompt_tokens_details.cached_tokens is how much of the input hit
        # a provider prompt cache (billed at a heavy discount)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        # usage.cost (USD) is OpenRouter-specific; absent on other stacks
        cost = getattr(usage, "cost", None)

        totals = self.usage_totals
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["cached_tokens"] += cached
        totals["cost"] += cost or 0.0
        totals["requests"] += 1

        per = self.usage_by_model.setdefault(model, {
            "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "cost": 0.0, "requests": 0,
        })
        per["prompt_tokens"] += prompt
        per["completion_tokens"] += completion
        per["cached_tokens"] += cached
        per["cost"] += cost or 0.0
        per["requests"] += 1

    def get_usage(self):
        """Token usage across all calls made by this agent."""
        return dict(self.usage_totals)

    def get_usage_by_model(self):
        """Per-model usage breakdown across this agent's calls."""
        return {model: dict(usage) for model, usage in self.usage_by_model.items()}

    def _append_transcript(self, original_input: str, output: str,
                           before: dict = None, before_by_model: dict = None) -> None:
        """Record the full, uncompressed input and final answer locally.

        When ``before`` snapshots are supplied (the normal path from
        run()), the entry records the usage *delta* of this run so the
        transcript can be summed per model without double-counting.
        """
        if not self.transcript_enabled:
            return
        try:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": self.config['openrouter']['model'],
                "input": original_input,
                "output": output,
            }
            if before is None:
                entry["usage"] = self.get_usage()
            else:
                after = self.get_usage()
                after_by_model = self.get_usage_by_model()
                entry["usage"] = {
                    key: after[key] - before.get(key, 0) for key in after
                }
                entry["by_model"] = {
                    model: {
                        key: usage[key] - before_by_model.get(model, {}).get(key, 0)
                        for key in usage
                    }
                    for model, usage in after_by_model.items()
                }
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # A failed transcript must never fail the run itself
            if not self.silent:
                print(f"⚠️  Could not write transcript: {e}")

    def _append_compaction_transcript(self, dropped_messages: list) -> None:
        """Persist turns about to be folded into a compaction digest.

        The wire copy of history is compressed, but every dropped turn
        lands here verbatim first - compaction never loses data.
        """
        if not self.transcript_enabled or not dropped_messages:
            return
        try:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": "compaction",
                "dropped": [self._message_snapshot(m) for m in dropped_messages],
            }
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            if not self.silent:
                print(f"⚠️  Could not write compaction transcript: {e}")

    @staticmethod
    def _message_snapshot(message: dict) -> dict:
        """JSON-serializable copy of a message, tolerant of SDK tool-call
        objects that json.dumps cannot serialize directly."""
        snapshot = {"role": message.get("role")}
        if message.get("content") is not None:
            snapshot["content"] = message.get("content")
        if message.get("name"):
            snapshot["name"] = message.get("name")
        if message.get("tool_calls"):
            snapshot["tool_calls"] = [
                {
                    "name": _tool_call_name(tc),
                    "arguments": _tool_call_arguments(tc),
                }
                for tc in message["tool_calls"]
            ]
        return snapshot

    def compact_history(self, messages: list) -> int:
        """Sticky in-place compaction of a live conversation. Returns the
        number of turns folded into the digest (they are persisted to the
        transcript first). Between compaction events the message list only
        grows by appends, so the prompt prefix stays stable and provider
        prompt caches can hit."""
        dropped = self.budget.compact_history(messages)
        if dropped:
            self._append_compaction_transcript(dropped)
        return len(dropped)

    
    def handle_tool_call(self, tool_call):
        """Handle a tool call and return the result message"""
        # Resolve upfront so the error path below can reference it
        tool_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
        try:
            # Extract tool name and arguments
            tool_args = json.loads(tool_call.function.arguments)

            # Call appropriate tool from tool_mapping
            if tool_name in self.tool_mapping:
                tool_result = self.tool_mapping[tool_name](**tool_args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            # Return tool result message
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            }
        
        except Exception as e:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps({"error": f"Tool execution failed: {str(e)}"})
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with context and memory if available."""
        prompt = self.config['system_prompt']

        if self.context_aware and self.project_context:
            # Add context section
            context_section = "\n\n## Project Context\n\n"
            context_section += self.project_context.get_formatted_context()

            # Add available tools reminder
            context_section += "\n\n## Available Tools\n"
            for tool_name in self.tool_mapping.keys():
                context_section += f"- {tool_name}\n"

            prompt += context_section

        # Memory index + recent log so the agent starts every run already
        # knowing what it knows (the wiki is the compounding artifact)
        if self.memory_store is not None:
            try:
                prompt += self.memory_store.get_prompt_section()
            except Exception as e:
                if not self.silent:
                    print(f"⚠️  Could not inject memory: {e}")

        return prompt
    
    def run(self, user_input: str, response_format: dict = None):
        """Run the agent with user input and return FULL conversation content.

        Overlong inputs are compressed for the wire, but the original text
        is preserved verbatim in the local transcript (logs/transcript.jsonl)
        so compression never loses data.

        ``response_format`` (see ``structured_output``) constrains the
        model to schema-valid JSON for callers like the orchestrator's
        triage and the planner; unsupported providers degrade to plain
        text automatically.
        """
        system_prompt = self._build_system_prompt()
        original_input = user_input
        before = self.get_usage()
        before_by_model = self.get_usage_by_model()
        result = self._run_loop(
            self.budget.compress_user_input(user_input), system_prompt, response_format
        )
        self._append_transcript(original_input, result, before, before_by_model)
        return result

    def _run_loop(self, user_input: str, system_prompt: str, response_format: dict = None):
        """The agentic loop itself. Returns the full response content."""
        # Initialize messages with system prompt and user input
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # Track all assistant responses for full content capture
        full_response_content = []
        
        # Implement agentic loop from OpenRouter docs
        max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if not self.silent:
                print(f"🔄 Agent iteration {iteration}/{max_iterations}")

            # Sticky compaction: when the history crosses the budget it is
            # rewritten in place (once), so every other iteration only
            # appends and the prompt prefix stays byte-stable between
            # compaction events - provider prompt caches can then hit.
            self.compact_history(messages)

            # Call LLM
            response = self.call_llm(messages, response_format)

            # Add the response to messages. Only include tool_calls when the
            # model actually made some: a null tool_calls field is rejected
            # by several providers.
            assistant_message = response.choices[0].message
            message_dict = {"role": "assistant", "content": assistant_message.content}
            if assistant_message.tool_calls:
                message_dict["tool_calls"] = assistant_message.tool_calls
            messages.append(message_dict)

            # Capture assistant content for full response
            if assistant_message.content:
                full_response_content.append(assistant_message.content)

            # Check if there are tool calls
            if assistant_message.tool_calls:
                if not self.silent:
                    print(f"🔧 Agent making {len(assistant_message.tool_calls)} tool call(s)")
                # Handle each tool call
                for tool_call in assistant_message.tool_calls:
                    if not self.silent:
                        print(f"   📞 Calling tool: {tool_call.function.name}")
                    tool_result = self.handle_tool_call(tool_call)
                    messages.append(tool_result)

                    # Check if this was the task completion tool
                    if tool_call.function.name == "mark_task_complete":
                        if not self.silent:
                            print("✅ Task completion tool called - exiting loop")
                        # Return FULL conversation content, not just completion message
                        return "\n\n".join(full_response_content)
            else:
                # A response with no tool calls is the model's final answer;
                # looping again would re-send the same conversation and
                # duplicate the response until max_iterations.
                if not self.silent:
                    print("💭 Agent responded without tool calls - task complete")
                break

        # Loop ended: either a final answer or max iterations reached
        return "\n\n".join(full_response_content) if full_response_content else "Maximum iterations reached. The agent may be stuck in a loop."