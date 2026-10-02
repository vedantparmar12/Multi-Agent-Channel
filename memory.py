"""Persistent agent memory as a markdown wiki (Karpathy's LLM-wiki pattern).

Layout, all bootstrapped on first use::

    memory/
      wiki/         # topic/entity pages, written by the agent itself
        <slug>.md
      index.md      # catalog: every page with a one-line summary + date
      log.md        # append-only timeline: "## [YYYY-MM-DD] action | title"

Deliberately simple, per the pattern that actually works in practice:

- plain markdown files, no database, no embeddings, no background jobs
- ``index.md`` is the navigation layer: injected into the system prompt so
  the agent knows what it already knows before answering
- ``log.md`` entries use a consistent ``## [date] action | title`` prefix so
  the timeline is parseable with plain grep: ``grep "^## \\[" memory/log.md``
- the agent files knowledge itself via the save_memory_page tool; the
  harness appends query entries automatically
"""

import re
import threading
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# Orchestrator agents run in separate threads and share the memory
# directory; index/log updates are read-modify-write, so serialize them.
# RLock because save_page calls append_log while already holding it.
_WRITE_LOCK = threading.RLock()

LOG_ACTIONS = ("query", "save", "read", "lint")
LOG_ENTRY_RE = re.compile(r"^## \[(\d{4}-\d{2}-\d{2})\] (\w+) \| (.+)$", re.MULTILINE)


def slugify(title: str) -> str:
    """Turn a page title into a safe filename slug.

    Memory pages are addressed by LLM-provided titles, so the slug must
    never carry path separators or traversal sequences.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80] or "untitled"


class MemoryStore:
    """Markdown-wiki memory rooted at ``directory`` (default: ./memory)."""

    def __init__(self, directory: Optional[str] = None, root: Optional[Path] = None):
        base = Path(root) if root else Path.cwd()
        self.base = (base / (directory or "memory")).resolve()
        self.wiki_dir = self.base / "wiki"
        self.index_path = self.base / "index.md"
        self.log_path = self.base / "log.md"
        self._bootstrap()

    def _bootstrap(self) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(
                "# Memory Index\n\nPages the agent has filed. Read a page with "
                "read_memory_page(title).\n",
                encoding="utf-8",
            )
        if not self.log_path.exists():
            self.log_path.write_text("# Memory Log\n", encoding="utf-8")

    # -- reading ----------------------------------------------------------

    def get_index(self) -> str:
        return self.index_path.read_text(encoding="utf-8")

    def get_log_tail(self, entries: int = 5) -> List[str]:
        """Most recent log lines, newest last (grep '^## \\[' equivalent)."""
        matches = LOG_ENTRY_RE.findall(self.log_path.read_text(encoding="utf-8"))
        return [f"[{d}] {action} | {title}" for d, action, title in matches[-entries:]]

    def read_page(self, title: str) -> Optional[str]:
        path = self.wiki_dir / f"{slugify(title)}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_pages(self) -> List[str]:
        return sorted(p.stem for p in self.wiki_dir.glob("*.md"))

    def list_page_titles(self) -> List[str]:
        """Human-readable page titles, parsed from the index catalog."""
        index = self.index_path.read_text(encoding="utf-8")
        return re.findall(r"^- \[(.+?)\]\(wiki/.+?\.md\)", index, re.MULTILINE)

    # -- writing ----------------------------------------------------------

    def save_page(self, title: str, content: str) -> str:
        """Write a wiki page, refresh the index, and append a log entry.

        Returns the path of the saved page. The index entry is replaced in
        place when the page already exists (dedup), never duplicated.
        """
        slug = slugify(title)
        path = self.wiki_dir / f"{slug}.md"
        with _WRITE_LOCK:
            path.write_text(content.rstrip() + "\n", encoding="utf-8")
            self._upsert_index_entry(title, slug)
            self.append_log("save", title)
        return str(path)

    def _upsert_index_entry(self, title: str, slug: str) -> None:
        today = date.today().isoformat()
        entry = f"- [{title}](wiki/{slug}.md) - {today}"
        current = self.index_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"^- \[.*?\]\(wiki/{re.escape(slug)}\.md\).*$", re.MULTILINE)
        if pattern.search(current):
            self.index_path.write_text(pattern.sub(entry, current), encoding="utf-8")
        else:
            if not current.endswith("\n"):
                current += "\n"
            self.index_path.write_text(current + entry + "\n", encoding="utf-8")

    def append_log(self, action: str, title: str) -> None:
        """Append a parseable timeline entry: ``## [date] action | title``."""
        if action not in LOG_ACTIONS:
            action = "save"
        entry = f"## [{date.today().isoformat()}] {action} | {title[:100]}"
        with _WRITE_LOCK:
            current = self.log_path.read_text(encoding="utf-8")
            if not current.endswith("\n"):
                current += "\n"
            self.log_path.write_text(current + entry + "\n", encoding="utf-8")

    # -- prompt integration -------------------------------------------------

    def get_prompt_section(self, max_chars: int = 1500) -> str:
        """Context block injected into the system prompt: index + recent log.

        Capped so memory cannot silently grow the prompt without bound; the
        agent can always read specific pages via the read tool.
        """
        section = "\n\n## Agent Memory\n\nYou maintain a persistent wiki of "
        section += "knowledge across sessions. Save durable facts, decisions, and "
        section += "user preferences with save_memory_page; retrieve them with "
        section += "read_memory_page. File good answers back as pages instead of "
        section += "letting them vanish into chat history.\n\n"
        section += "### Known pages (index.md)\n"
        section += self.get_index()
        log_tail = self.get_log_tail()
        if log_tail:
            section += "\n### Recent activity (log.md)\n"
            section += "\n".join(log_tail)
        if len(section) > max_chars:
            section = section[:max_chars] + "\n[... memory index truncated ...]"
        return section
