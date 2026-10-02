"""Tool for loading and managing project context."""

from typing import Dict, Any
from .base_tool import BaseTool
from context import ContextLoader

class ContextTool(BaseTool):
    """Tool that provides project context to agents."""
    
    def __init__(self, config: dict):
        self.config = config
        self.context_loader = ContextLoader()
    
    @property
    def name(self) -> str:
        return "load_project_context"
    
    @property
    def description(self) -> str:
        return "Load project context including CLAUDE.md rules, planning docs, tasks, and examples"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "include_examples": {
                    "type": "boolean",
                    "description": "Whether to include code examples in the context",
                    "default": True
                },
                "output_format": {
                    "type": "string",
                    "description": "Format for the context: 'full' or 'summary'",
                    "enum": ["full", "summary"],
                    "default": "full"
                }
            },
            "required": []
        }

    def execute(self, include_examples: bool = True, output_format: str = "full") -> Dict[str, Any]:
        """Load and return project context.

        Args:
            include_examples: Whether to include code examples
            output_format: How to format the context

        Returns:
            Dictionary containing project context
        """
        try:
            # Load full context
            context = self.context_loader.load_project_context()

            if output_format == "summary":
                # Return summary view
                return {
                    "status": "success",
                    "claude_rules": len(context.claude_rules) > 0,
                    "planning": context.planning is not None,
                    "tasks": context.tasks is not None,
                    "examples_count": len(context.examples),
                    "prps_count": len(context.prps),
                    "formatted_context": context.get_formatted_context()[:500] + "..."
                }
            else:
                # Return full context
                result = {
                    "status": "success",
                    "claude_rules": context.claude_rules,
                    "planning": context.planning,
                    "tasks": context.tasks,
                    "prps": list(context.prps.keys())
                }
                
                if include_examples:
                    result["examples"] = list(context.examples.keys())
                    result["example_snippets"] = {
                        name: content[:200] + "..." 
                        for name, content in context.examples.items()
                    }
                
                return result
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load context: {str(e)}"
            }