from .base_tool import BaseTool, resolve_safe_path
import os

class WriteFileTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    
    def execute(self, path: str, content: str) -> dict:
        try:
            # Sandbox the path to the project root
            try:
                abs_path = resolve_safe_path(path)
            except ValueError as e:
                return {"error": str(e)}

            # Create parent directories if needed
            parent_dir = abs_path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)

            # Write file atomically using temporary file
            temp_path = parent_dir / (abs_path.name + '.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # os.replace works on all platforms even when the
                # destination already exists; os.rename does not on Windows
                os.replace(temp_path, abs_path)

                return {
                    "path": str(abs_path),
                    "bytes_written": len(content.encode('utf-8')),
                    "success": True,
                    "message": f"Successfully wrote to {path}"
                }

            except Exception:
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise

        except PermissionError:
            return {"error": f"Permission denied writing to file: {path}"}
        except OSError as e:
            return {"error": f"OS error writing file: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}