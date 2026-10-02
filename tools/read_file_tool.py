from .base_tool import BaseTool, resolve_safe_path

class ReadFileTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the complete contents of a file from the file system. Handles various text encodings and provides detailed error messages if the file cannot be read."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                },
                "head": {
                    "type": "integer",
                    "description": "If provided, returns only the first N lines of the file"
                },
                "tail": {
                    "type": "integer",
                    "description": "If provided, returns only the last N lines of the file"
                }
            },
            "required": ["path"]
        }
    
    def execute(self, path: str, head: int = None, tail: int = None) -> dict:
        try:
            # Validate parameters
            if head is not None and tail is not None:
                return {"error": "Cannot specify both head and tail parameters"}

            # Sandbox the path to the project root
            try:
                safe_path = resolve_safe_path(path)
            except ValueError as e:
                return {"error": str(e)}

            # Check if file exists
            if not safe_path.exists():
                return {"error": f"File not found: {path}"}

            # Check if it's actually a file (not a directory)
            if not safe_path.is_file():
                return {"error": f"Path is not a file: {path}"}

            # Read file with appropriate method
            if head is not None:
                # Read first N lines
                with safe_path.open('r', encoding='utf-8') as f:
                    lines = []
                    for i in range(head):
                        line = f.readline()
                        if not line:  # EOF reached
                            break
                        lines.append(line)
                    content = ''.join(lines).rstrip('\n')
            elif tail is not None:
                # Read last N lines
                with safe_path.open('r', encoding='utf-8') as f:
                    lines = f.readlines()
                    content = ''.join(lines[-tail:]).rstrip('\n') if lines else ""
            else:
                # Read entire file
                content = safe_path.read_text(encoding='utf-8')

            return {
                "path": str(safe_path),
                "content": content,
                "success": True
            }
        
        except UnicodeDecodeError as e:
            return {"error": f"Failed to decode file as UTF-8: {str(e)}"}
        except PermissionError:
            return {"error": f"Permission denied reading file: {path}"}
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}