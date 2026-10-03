import json
import re
import yaml
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any, Optional
from agent import OpenRouterAgent, structured_output

FALLBACK_QUESTION_TEMPLATES = [
    "Research comprehensive information about: {user_input}",
    "Analyze and provide insights about: {user_input}",
    "Find alternative perspectives on: {user_input}",
    "Verify and cross-check facts about: {user_input}",
]

DEFAULT_MAX_DYNAMIC_AGENTS = 8

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
        "num_agents": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["complexity", "num_agents"],
}

QUESTIONS_SCHEMA = {"type": "array", "items": {"type": "string"}}

DEFAULT_TRIAGE_PROMPT = """You are a request triage router for a multi-agent system.

Classify this request: "{user_input}"

- "simple" if one capable agent can answer it directly: a single factual
  question, a lookup, a calculation, chit-chat, a small code snippet.
- "complex" if it benefits from parallel agents covering different angles:
  research, comparisons, multi-facet analysis, "everything about X".

For complex requests also propose how many agents (2 to {max_agents})
the request actually deserves - do not pad: a two-sided comparison needs
2-3, a broad survey might justify more.

Respond with ONLY this JSON and nothing else:
{{"complexity": "simple"|"complex", "num_agents": <int>, "reasoning": "<one line>"}}"""


def build_fallback_questions(user_input: str, num_agents: int) -> List[str]:
    """Deterministic subtasks used when AI question generation fails.

    Cycles the templates so any agent count is covered (the previous
    hardcoded 4-item list caused an IndexError for parallel_agents > 4).
    """
    return [
        f"[Angle {i + 1}] " + FALLBACK_QUESTION_TEMPLATES[i % len(FALLBACK_QUESTION_TEMPLATES)].format(user_input=user_input)
        for i in range(num_agents)
    ]

class TaskOrchestrator:
    def __init__(self, config_path="config.yaml", silent=False):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_agents = self.config['orchestrator']['parallel_agents']
        self.task_timeout = self.config['orchestrator']['task_timeout']
        self.aggregation_strategy = self.config['orchestrator']['aggregation_strategy']
        self.silent = silent

        # Dynamic spawning: a cheap triage call classifies each request and
        # proposes the agent count, instead of always fanning out the fixed
        # configured number (simple requests skip the fan-out entirely)
        orchestrator_config = self.config['orchestrator']
        self.dynamic_enabled = orchestrator_config.get('dynamic', False)
        self.max_dynamic_agents = int(orchestrator_config.get('max_agents', DEFAULT_MAX_DYNAMIC_AGENTS))
        self.triage_prompt_template = orchestrator_config.get('triage_prompt', DEFAULT_TRIAGE_PROMPT)
        # Agent count actually used by the current/last orchestrate() run;
        # the progress display reads this instead of the static config value
        self.active_count = self.num_agents

        # Track agent progress
        self.agent_progress = {}
        self.agent_results = {}
        self.progress_lock = threading.Lock()
    
    def decompose_task(self, user_input: str, num_agents: int) -> List[str]:
        """Use AI to dynamically generate different questions based on user input"""
        
        # Create question generation agent
        question_agent = OpenRouterAgent(silent=True)
        
        # Get question generation prompt from config
        prompt_template = self.config['orchestrator']['question_generation_prompt']
        generation_prompt = prompt_template.format(
            user_input=user_input,
            num_agents=num_agents
        )
        
        # Remove task completion tool to avoid issues
        question_agent.tools = [tool for tool in question_agent.tools if tool.get('function', {}).get('name') != 'mark_task_complete']
        question_agent.tool_mapping = {name: func for name, func in question_agent.tool_mapping.items() if name != 'mark_task_complete'}
        
        try:
            # Get AI-generated questions (schema-constrained when the
            # provider supports structured outputs, plain text otherwise)
            response = question_agent.run(
                generation_prompt,
                response_format=structured_output("questions", QUESTIONS_SCHEMA),
            )
            questions = self._parse_questions(response)
        except Exception:
            # Any failure (API error, malformed output) falls back to
            # deterministic subtasks instead of crashing orchestration
            questions = None

        if questions is not None and len(questions) == num_agents:
            return questions
        return build_fallback_questions(user_input, num_agents)

    @staticmethod
    def _parse_questions(response: str) -> Optional[List[str]]:
        """Parse a JSON array of questions, tolerating prose/markdown wrappers."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

        try:
            questions = json.loads(text)
        except json.JSONDecodeError:
            # Models often wrap the JSON array in prose; grab the array itself
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match is None:
                return None
            try:
                questions = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        if not isinstance(questions, list):
            return None
        if not all(isinstance(q, str) and q.strip() for q in questions):
            return None
        return questions
    
    def triage_request(self, user_input: str) -> Dict[str, Any]:
        """Classify a request and propose an agent count for it.

        Returns {"complexity": "simple"|"complex", "num_agents": int,
        "reasoning": str}. On any failure, falls back to the configured
        static agent count so dynamic mode degrades to the old behavior
        rather than erroring.
        """
        fallback = {
            "complexity": "complex",
            "num_agents": self.num_agents,
            "reasoning": "triage unavailable - using configured agent count",
        }
        try:
            triage_agent = OpenRouterAgent(silent=True)
            # One call, no tools: triage is routing, not research
            triage_agent.tools = []
            triage_agent.tool_mapping = {}

            prompt = self.triage_prompt_template.format(
                user_input=user_input,
                max_agents=self.max_dynamic_agents,
            )
            parsed = self._parse_json_object(
                triage_agent.run(
                    prompt,
                    response_format=structured_output("triage", TRIAGE_SCHEMA),
                )
            )
        except Exception:
            return fallback

        if parsed is None:
            return fallback

        complexity = parsed.get("complexity")
        if complexity not in ("simple", "complex"):
            return fallback

        # Clamp the model-proposed count so it can't over-decompose trivia
        # or fan out absurdly
        try:
            num_agents = int(parsed.get("num_agents", self.num_agents))
        except (TypeError, ValueError):
            num_agents = self.num_agents
        num_agents = max(1, min(num_agents, self.max_dynamic_agents))

        return {
            "complexity": complexity,
            "num_agents": num_agents,
            "reasoning": str(parsed.get("reasoning", ""))[:200],
        }

    @staticmethod
    def _parse_json_object(response: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from a model response, tolerating prose and
        markdown fences."""
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

        candidates = [text]
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def update_agent_progress(self, agent_id: int, status: str, result: str = None):
        """Thread-safe progress tracking"""
        with self.progress_lock:
            self.agent_progress[agent_id] = status
            if result is not None:
                self.agent_results[agent_id] = result
    
    def run_agent_parallel(self, agent_id: int, subtask: str) -> Dict[str, Any]:
        """
        Run a single agent with the given subtask.
        Returns result dictionary with agent_id, status, and response.
        """
        try:
            self.update_agent_progress(agent_id, "PROCESSING...")

            # Use simple agent like in main.py
            agent = OpenRouterAgent(silent=True)
            try:
                start_time = time.time()
                response = agent.run(subtask)
                execution_time = time.time() - start_time
            finally:
                # Workers are created per subtask; release their MCP
                # subprocesses (if any) right away instead of at exit
                close = getattr(agent, "close", None)
                if callable(close):
                    close()
            
            self.update_agent_progress(agent_id, "COMPLETED", response)
            
            return {
                "agent_id": agent_id,
                "status": "success", 
                "response": response,
                "execution_time": execution_time
            }
            
        except Exception as e:
            # Simple error handling
            return {
                "agent_id": agent_id,
                "status": "error",
                "response": f"Error: {str(e)}",
                "execution_time": 0
            }
    
    def aggregate_results(self, agent_results: List[Dict[str, Any]]) -> str:
        """
        Combine results from all agents into a comprehensive final answer.
        Uses the configured aggregation strategy.
        """
        successful_results = [r for r in agent_results if r["status"] == "success"]
        
        if not successful_results:
            return "All agents failed to provide results. Please try again."
        
        # Extract responses for aggregation
        responses = [r["response"] for r in successful_results]
        
        if self.aggregation_strategy == "consensus":
            return self._aggregate_consensus(responses, successful_results)
        else:
            # Default to consensus
            return self._aggregate_consensus(responses, successful_results)
    
    def _aggregate_consensus(self, responses: List[str], _results: List[Dict[str, Any]]) -> str:
        """
        Use one final AI call to synthesize all agent responses into a coherent answer.
        """
        if len(responses) == 1:
            return responses[0]
        
        # Create synthesis agent to combine all responses
        synthesis_agent = OpenRouterAgent(silent=True)
        
        # Build agent responses section
        agent_responses_text = ""
        for i, response in enumerate(responses, 1):
            agent_responses_text += f"=== AGENT {i} RESPONSE ===\n{response}\n\n"
        
        # Get synthesis prompt from config and format it
        synthesis_prompt_template = self.config['orchestrator']['synthesis_prompt']
        synthesis_prompt = synthesis_prompt_template.format(
            num_responses=len(responses),
            agent_responses=agent_responses_text
        )
        
        # Completely remove all tools from synthesis agent to force direct response
        synthesis_agent.tools = []
        synthesis_agent.tool_mapping = {}
        
        # Get the synthesized response
        try:
            final_answer = synthesis_agent.run(synthesis_prompt)
            return final_answer
        except Exception as e:
            # Log the error for debugging
            print(f"\n🚨 SYNTHESIS FAILED: {str(e)}")
            print("📋 Falling back to concatenated responses\n")
            # Fallback: if synthesis fails, concatenate responses
            combined = []
            for i, response in enumerate(responses, 1):
                combined.append(f"=== Agent {i} Response ===")
                combined.append(response)
                combined.append("")
            return "\n".join(combined)
    
    def get_progress_status(self) -> Dict[int, str]:
        """Get current progress status for all agents"""
        with self.progress_lock:
            return self.agent_progress.copy()
    
    def orchestrate(self, user_input: str):
        """
        Main orchestration method. Takes user input, decides how much
        firepower it needs (triage), delegates to agents, and returns the
        aggregated result.
        """
        # Reset progress tracking
        self.agent_progress = {}
        self.agent_results = {}

        # Route: simple requests get one agent; complex requests get a
        # model-proposed number of parallel agents
        num_agents = self.num_agents
        if self.dynamic_enabled:
            triage = self.triage_request(user_input)
            if not self.silent:
                print(f"🧭 Triage: {triage['complexity']}"
                      + (f" x{triage['num_agents']}" if triage['complexity'] == 'complex' else "")
                      + (f" - {triage['reasoning']}" if triage.get('reasoning') else ""))
            if triage["complexity"] == "simple":
                # One capable agent, no decomposition, no synthesis pass -
                # 1 API call path instead of N+2
                self.active_count = 1
                self.agent_progress = {0: "QUEUED"}
                result = self.run_agent_parallel(0, user_input)
                return result["response"]
            num_agents = triage["num_agents"]

        self.active_count = num_agents
        subtasks = self.decompose_task(user_input, num_agents)

        # Initialize progress tracking
        for i in range(num_agents):
            self.agent_progress[i] = "QUEUED"

        # Execute agents in parallel
        agent_results = []

        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self.run_agent_parallel, i, subtasks[i]): i
                for i in range(num_agents)
            }

            try:
                # Collect results as they complete
                for future in as_completed(future_to_agent, timeout=self.task_timeout):
                    agent_id = future_to_agent[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "agent_id": agent_id,
                            "status": "error",
                            "response": f"Agent {agent_id + 1} failed: {str(e)}",
                            "execution_time": 0
                        }
                    agent_results.append(result)
            except FuturesTimeoutError:
                # as_completed raises from the iterator itself, which sits
                # outside the inner try/except. Harvest whatever finished by
                # the deadline and mark the rest as timed out.
                for future, agent_id in future_to_agent.items():
                    if future.done() and not future.cancelled():
                        try:
                            agent_results.append(future.result())
                            continue
                        except Exception:
                            pass
                    future.cancel()
                    agent_results.append({
                        "agent_id": agent_id,
                        "status": "timeout",
                        "response": f"Agent {agent_id + 1} timed out after {self.task_timeout}s",
                        "execution_time": self.task_timeout
                    })
        
        # Sort results by agent_id for consistent output
        agent_results.sort(key=lambda x: x["agent_id"])
        
        # Aggregate results
        final_result = self.aggregate_results(agent_results)
        
        return final_result