"""Tests for the critical bug fixes: path sandboxing, tool safety, and
orchestrator fallback logic."""

import pytest

from tools.base_tool import resolve_safe_path
from tools.calculator_tool import CalculatorTool
from tools.read_file_tool import ReadFileTool
from tools.write_file_tool import WriteFileTool
from orchestrator import build_fallback_questions, TaskOrchestrator


class TestResolveSafePath:
    def test_allows_project_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = resolve_safe_path("src/main.py")
        assert resolved == (tmp_path / "src" / "main.py").resolve()

    def test_allows_absolute_path_inside_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = resolve_safe_path(str(tmp_path / "notes.txt"))
        assert resolved == (tmp_path / "notes.txt").resolve()

    def test_rejects_relative_escape(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_safe_path("../outside.txt")

    def test_rejects_absolute_escape(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_safe_path(str(tmp_path.parent / "outside.txt"))

    @pytest.mark.parametrize("sensitive", [".env", "config.yaml", "server.key"])
    def test_rejects_credential_files(self, tmp_path, monkeypatch, sensitive):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_safe_path(sensitive)


class TestFileToolSandbox:
    def test_write_rejects_escape_outside_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = WriteFileTool({})
        result = tool.execute(str(tmp_path.parent / "escaped.txt"), "malicious")
        assert result.get("error")
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_write_rejects_credential_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = WriteFileTool({})
        result = tool.execute(".env", "stolen-key=abc")
        assert result.get("error")
        assert not (tmp_path / ".env").exists()

    def test_write_then_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_tool = WriteFileTool({})
        result = write_tool.execute("notes/inner.txt", "hello world")
        assert result.get("success") is True

        read_tool = ReadFileTool({})
        assert read_tool.execute("notes/inner.txt")["content"] == "hello world"

    def test_write_overwrites_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tool = WriteFileTool({})
        tool.execute("file.txt", "first")
        result = tool.execute("file.txt", "second")
        assert result.get("success") is True
        assert (tmp_path / "file.txt").read_text() == "second"
        # No leftover temp files from the atomic write
        assert not (tmp_path / "file.txt.tmp").exists()

    def test_read_rejects_credential_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("SECRET=1")
        tool = ReadFileTool({})
        result = tool.execute(".env")
        assert result.get("error")

    def test_read_rejects_escape_outside_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        try:
            tool = ReadFileTool({})
            assert tool.execute(str(outside)).get("error")
        finally:
            outside.unlink()


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        calc = CalculatorTool({})
        assert calc.execute("2 + 3 * 4")["result"] == 14

    def test_rejects_huge_exponent(self):
        calc = CalculatorTool({})
        result = calc.execute("9**9**9")
        assert result["success"] is False
        assert "Exponent too large" in result["error"]

    def test_allows_reasonable_exponent(self):
        calc = CalculatorTool({})
        assert calc.execute("2**10")["result"] == 1024


class TestOrchestratorFallback:
    def test_fallback_questions_match_agent_count(self):
        questions = build_fallback_questions("quantum computing", 7)
        assert len(questions) == 7
        assert all(isinstance(q, str) and "quantum computing" in q for q in questions)

    def test_fallback_questions_cover_small_counts(self):
        assert len(build_fallback_questions("topic", 1)) == 1
        assert len(build_fallback_questions("topic", 3)) == 3

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ('```json\n["a", "b"]\n```', ["a", "b"]),
            ('Here are the questions:\n["a", "b"]\nHope that helps!', ["a", "b"]),
            ('not json at all', None),
            ('{"key": "value"}', None),
            ('["a", 42]', None),
            ('[]', []),
        ],
    )
    def test_parse_questions_handles_model_output(self, raw, expected):
        assert TaskOrchestrator._parse_questions(raw) == expected
