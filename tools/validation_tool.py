"""Tool for running validation and tests on code."""

import re
import subprocess
from typing import Dict, Any
from pathlib import Path
from .base_tool import BaseTool, resolve_safe_path

class ValidationTool(BaseTool):
    """Tool that runs linting, type checking, and tests."""

    def __init__(self, config: dict):
        self.config = config
        self.validation_config = config.get('validation', {})
        self.timeout = self.validation_config.get('timeout', 300)
    
    @property
    def name(self) -> str:
        return "run_validation"
    
    @property
    def description(self) -> str:
        return "Run code validation including linting (ruff), type checking (mypy), and tests (pytest)"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "validation_type": {
                    "type": "string",
                    "description": "Type of validation to run",
                    "enum": ["lint", "typecheck", "test", "all"],
                    "default": "all"
                },
                "target": {
                    "type": "string",
                    "description": "File or directory to validate",
                    "default": "."
                },
                "fix": {
                    "type": "boolean",
                    "description": "Attempt to auto-fix issues (only for linting)",
                    "default": True
                }
            },
            "required": ["validation_type"]
        }
    
    def execute(self, validation_type: str = "all", target: str = ".", fix: bool = True) -> Dict[str, Any]:
        """Run validation on code.
        
        Args:
            validation_type: Type of validation to run
            target: File or directory to validate
            fix: Whether to auto-fix linting issues
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "status": "success",
            "validations": {}
        }
        
        try:
            if validation_type in ["lint", "all"]:
                results["validations"]["lint"] = self._run_linting(target, fix)
                
            if validation_type in ["typecheck", "all"]:
                results["validations"]["typecheck"] = self._run_typecheck(target)
                
            if validation_type in ["test", "all"]:
                results["validations"]["test"] = self._run_tests(target)
            
            # Determine overall status
            all_passed = all(
                v.get("success", False) 
                for v in results["validations"].values()
            )
            results["status"] = "success" if all_passed else "failed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Validation failed: {str(e)}"
            }
    
    def _run_command(self, cmd: list) -> Dict[str, Any]:
        """Run a validation command with a bounded lifetime.

        Without a timeout a hung ruff/mypy/pytest process blocks the agent
        loop forever.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"'{cmd[0]}' timed out after {self.timeout}s"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"'{cmd[0]}' not found on PATH"
            }

    def _resolve_target(self, target: str) -> Path:
        # Targets are LLM-chosen; keep them inside the project root
        return resolve_safe_path(target)

    def _run_linting(self, target: str, fix: bool) -> Dict[str, Any]:
        """Run ruff linting."""
        try:
            safe_target = self._resolve_target(target)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        cmd = ["ruff", "check", str(safe_target)]
        if fix:
            cmd.append("--fix")

        result = self._run_command(cmd)
        if "output" in result:
            result["fixed"] = fix and "fixed" in result["output"].lower()
        return result

    def _run_typecheck(self, target: str) -> Dict[str, Any]:
        """Run mypy type checking."""
        try:
            safe_target = self._resolve_target(target)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        return self._run_command(["mypy", str(safe_target), "--ignore-missing-imports"])

    def _run_tests(self, target: str) -> Dict[str, Any]:
        """Run pytest tests."""
        try:
            safe_target = self._resolve_target(target)
            tests_dir = resolve_safe_path("tests")
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Determine test path
        if target == ".":
            test_path = "tests/"
        elif tests_dir == safe_target or tests_dir in safe_target.parents:
            test_path = str(safe_target)
        else:
            # Find corresponding test file
            test_path = str(tests_dir / f"test_{safe_target.name}")

        result = self._run_command(["pytest", test_path, "-v", "--tb=short"])
        if "output" in result:
            # pytest summary lines look like "3 passed, 1 failed in 0.5s";
            # match each count independently so "5 passed in 0.1s" is counted
            output = result["output"]
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            result["passed"] = int(passed_match.group(1)) if passed_match else 0
            result["failed"] = int(failed_match.group(1)) if failed_match else 0
        return result