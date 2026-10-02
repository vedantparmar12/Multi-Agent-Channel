from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

# Tools execute on LLM-chosen paths, so file access must stay inside the
# project root and away from files that hold credentials.
SENSITIVE_FILE_NAMES = {".env", "config.yaml"}
SENSITIVE_SUFFIXES = {".key"}


def resolve_safe_path(path: str, root: Path = None) -> Path:
    """Resolve a tool path within the project root, rejecting escapes
    and credential files.

    Raises ValueError on any violation; callers are expected to surface
    the message to the model as a tool error.
    """
    base = (root or Path.cwd()).resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    resolved = resolved.resolve()

    if not resolved.is_relative_to(base):
        raise ValueError(
            f"Access denied: '{path}' resolves outside the project root ({base})"
        )
    if resolved.name in SENSITIVE_FILE_NAMES or resolved.suffix in SENSITIVE_SUFFIXES:
        raise ValueError(f"Access denied: '{resolved.name}' may contain secrets")
    return resolved


class BaseTool(ABC):
    """Base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for OpenRouter function calling"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for OpenRouter"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """OpenRouter function parameters schema"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    def to_openrouter_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenRouter function schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }