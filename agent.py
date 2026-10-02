import json
import sys
import yaml
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from tools import discover_tools
from token_budget import TokenBudget
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

        # Usage accumulated across all API calls in this agent's lifetime
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

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
    
    
    def call_llm(self, messages):
        """Make an OpenRouter API call with retries and model fallback.

        The token-budget proxy runs first (messages are truncated/compacted
        just before hitting the wire), then each model is attempted with
        backoff on transient errors; non-transient errors fail over to the
        next model immediately since retrying cannot fix them.
        """
        prepared = self.budget.prepare_messages(messages)
        models = [self.config['openrouter']['model']] + self.fallback_models
        last_error = None

        for model in models:
            try:
                response = retry_call(
                    lambda m=model: self._create_completion(m, prepared),
                    attempts=self.retry_attempts,
                    base_delay=self.retry_base_delay,
                )
                self._record_usage(response)
                return response
            except Exception as e:
                last_error = e
                if not self.silent:
                    print(f"⚠️  model '{model}' unavailable: {e}")

        raise Exception(f"LLM call failed: {last_error}")

    def _create_completion(self, model, messages):
        request_kwargs = {
            "model": model,
            "messages": messages,
        }
        # An empty tools array is rejected by many providers
        if self.tools:
            request_kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**request_kwargs)

    def _record_usage(self, response):
        """Accumulate real usage numbers returned by the API."""
        usage = getattr(response, "usage", None)
        if usage:
            self.usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self.usage_totals["requests"] += 1

    def get_usage(self):
        """Token usage across all calls made by this agent."""
        return dict(self.usage_totals)

    def _append_transcript(self, original_input: str, output: str) -> None:
        """Record the full, uncompressed input and final answer locally."""
        if not self.transcript_enabled:
            return
        try:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input": original_input,
                "output": output,
                "usage": self.get_usage(),
            }
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # A failed transcript must never fail the run itself
            if not self.silent:
                print(f"⚠️  Could not write transcript: {e}")
    
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
    
    def run(self, user_input: str):
        """Run the agent with user input and return FULL conversation content.

        Overlong inputs are compressed for the wire, but the original text
        is preserved verbatim in the local transcript (logs/transcript.jsonl)
        so compression never loses data.
        """
        system_prompt = self._build_system_prompt()
        original_input = user_input
        result = self._run_loop(self.budget.compress_user_input(user_input), system_prompt)
        self._append_transcript(original_input, result)
        return result

    def _run_loop(self, user_input: str, system_prompt: str):
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
            
            # Call LLM
            response = self.call_llm(messages)

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