import json
import yaml
from openai import OpenAI
from tools import discover_tools
from token_budget import TokenBudget

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
        """Make OpenRouter API call with tools.

        The token-budget proxy runs here: messages are truncated/compacted
        just before hitting the wire, so nothing is paid for twice.
        """
        try:
            prepared = self.budget.prepare_messages(messages)
            request_kwargs = {
                "model": self.config['openrouter']['model'],
                "messages": prepared,
            }
            # An empty tools array is rejected by many providers
            if self.tools:
                request_kwargs["tools"] = self.tools
            response = self.client.chat.completions.create(**request_kwargs)
            self._record_usage(response)
            return response
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")

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
        """Run the agent with user input and return FULL conversation content"""
        # Build system prompt with context and memory if available
        system_prompt = self._build_system_prompt()

        # Compress overlong user input before its first send: the original
        # text stays in the local transcript, only the wire copy shrinks
        user_input = self.budget.compress_user_input(user_input)

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