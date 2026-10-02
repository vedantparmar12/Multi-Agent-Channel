"""Agent-facing tools for the persistent markdown-wiki memory."""

from typing import Dict, Any, Optional
from .base_tool import BaseTool
from memory import MemoryStore


class SaveMemoryPageTool(BaseTool):
    """Files knowledge into the persistent wiki (writes page + updates index)."""

    def __init__(self, config: dict):
        self.config = config
        memory_config = config.get('memory', {}) if isinstance(config, dict) else {}
        self.store = MemoryStore(directory=memory_config.get('directory', 'memory'))

    @property
    def name(self) -> str:
        return "save_memory_page"

    @property
    def description(self) -> str:
        return (
            "Save durable knowledge to the persistent agent memory wiki. Use for "
            "facts, decisions, user preferences, and useful answers worth keeping "
            "across sessions. Re-saving an existing title updates that page."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short page title, e.g. 'User preferences' or 'REST API design decisions'"
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content of the page"
                }
            },
            "required": ["title", "content"]
        }

    def execute(self, title: str, content: str) -> Dict[str, Any]:
        try:
            if not title.strip() or not content.strip():
                return {"error": "Both title and content are required"}
            path = self.store.save_page(title, content)
            return {
                "status": "success",
                "path": path,
                "message": f"Memory page '{title}' saved and indexed"
            }
        except Exception as e:
            return {"error": f"Failed to save memory page: {str(e)}"}


class ReadMemoryPageTool(BaseTool):
    """Retrieves a page from the persistent wiki by title."""

    def __init__(self, config: dict):
        self.config = config
        memory_config = config.get('memory', {}) if isinstance(config, dict) else {}
        self.store = MemoryStore(directory=memory_config.get('directory', 'memory'))

    @property
    def name(self) -> str:
        return "read_memory_page"

    @property
    def description(self) -> str:
        return (
            "Read a page from the agent memory wiki by its title. Check the "
            "memory index in your context for available page titles first."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the memory page to read"
                }
            },
            "required": ["title"]
        }

    def execute(self, title: str) -> Dict[str, Any]:
        try:
            content = self.store.read_page(title)
            if content is None:
                available = ", ".join(self.store.list_page_titles()) or "none yet"
                return {
                    "error": f"No memory page titled '{title}'",
                    "available_pages": available
                }
            self.store.append_log("read", title)
            return {"status": "success", "title": title, "content": content}
        except Exception as e:
            return {"error": f"Failed to read memory page: {str(e)}"}
