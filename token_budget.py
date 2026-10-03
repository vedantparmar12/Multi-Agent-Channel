"""Token budget middleware: the proxy between the agent loop and the API.

Every request the agent sends passes through ``prepare_messages``. In an
agentic loop the full conversation is re-sent on every iteration, so tool
outputs and old turns compound the input cost each round. This module
shrinks what actually goes over the wire:

- tool result contents are truncated to a character limit
- when the estimated token count exceeds the budget, older turns are
  replaced by a compact digest (system prompt, original request, and the
  most recent turns are always kept verbatim)
- very long user inputs are compressed extractively before their first send

Everything here is deterministic and local: no extra LLM calls, no
embeddings. Token counts are estimates (~4 chars per token), which is
accurate enough for budgeting decisions; exact usage is reported
separately from the API responses.
"""

import re
from typing import Dict, List

DEFAULTS = {
    # Estimated-token budget for the whole message list before compaction
    "history_budget": 24000,
    # Hard character cap per tool result sent back to the model
    "tool_result_limit": 2000,
    # User inputs longer than this (in characters) are compressed before sending
    "input_compression_threshold": 4000,
    # Target size after input compression, as a fraction of the threshold
    "input_compression_ratio": 0.5,
    # Turns kept verbatim at the end of the conversation during compaction
    "recent_turns_kept": 6,
}

# Rough heuristic: English prose averages ~4 characters per token
CHARS_PER_TOKEN = 4

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _tool_call_name(tc) -> str:
    """Name of a tool call, tolerant of SDK objects and plain dicts."""
    fn = getattr(tc, "function", None)
    if fn is None and isinstance(tc, dict):
        fn = tc.get("function", {})
    return getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None) or "?"


def _tool_call_arguments(tc) -> str:
    """Argument text of a tool call, tolerant of SDK objects and plain dicts."""
    fn = getattr(tc, "function", None)
    if fn is None and isinstance(tc, dict):
        fn = tc.get("function", {})
    args = getattr(fn, "arguments", None)
    if args is None and isinstance(fn, dict):
        args = fn.get("arguments") or ""
    return args or ""


def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for message in messages:
        total += estimate_tokens(message.get("content") or "")
        for tool_call in message.get("tool_calls") or []:
            total += estimate_tokens(_tool_call_arguments(tool_call))
    return total


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _sentence_score(sentence: str) -> float:
    """Information density: share of content words, boosted by numbers and
    capitalized entities (names, URLs, code-ish tokens)."""
    words = re.findall(r"[A-Za-z0-9_'@./-]+", sentence)
    if not words:
        return 0.0
    content = [w for w in words if w.lower() not in STOPWORDS]
    score = len(content) / len(words)
    if re.search(r"\d", sentence):
        score += 0.15
    if re.search(r"\b[A-Z][a-z]+\b", sentence):
        score += 0.05
    return score


def compress_text(text: str, target_chars: int) -> str:
    """Extractively compress text to roughly ``target_chars``.

    Sentences are ranked by information density; the highest-scoring ones
    are kept (in original order) until the target is reached. The first
    sentence is always kept because it usually frames the request.
    """
    text = _collapse_whitespace(text)
    if len(text) <= target_chars:
        return text

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        # One giant unbroken blob: hard truncate at a word boundary
        return text[:target_chars].rsplit(" ", 1)[0] + " [...]"

    ranked = sorted(
        range(len(sentences)),
        key=lambda i: _sentence_score(sentences[i]),
        reverse=True,
    )
    keep = {0}
    size = len(sentences[0])
    for idx in ranked:
        if idx in keep:
            continue
        if size + len(sentences[idx]) > target_chars:
            continue
        keep.add(idx)
        size += len(sentences[idx]) + 1
        if size >= target_chars:
            break

    kept = [sentences[i] for i in sorted(keep)]
    result = " ".join(kept)
    if len(result) < len(text):
        result += "\n[... input compressed to save tokens ...]"
    return result


class TokenBudget:
    """Middleware applied to every outgoing request."""

    def __init__(self, config: Dict = None):
        config = config or {}
        self.history_budget = int(config.get("history_budget", DEFAULTS["history_budget"]))
        self.tool_result_limit = int(config.get("tool_result_limit", DEFAULTS["tool_result_limit"]))
        self.input_compression_threshold = int(
            config.get("input_compression_threshold", DEFAULTS["input_compression_threshold"])
        )
        self.input_compression_ratio = float(
            config.get("input_compression_ratio", DEFAULTS["input_compression_ratio"])
        )
        self.recent_turns_kept = int(config.get("recent_turns_kept", DEFAULTS["recent_turns_kept"]))
        # Set when prepare_messages actually shrinks something, for reporting
        self.last_report = {"tool_results_truncated": 0, "compacted": False}

    def compress_user_input(self, text: str) -> str:
        """Compress an overlong user prompt before its first send."""
        if len(text) <= self.input_compression_threshold:
            return text
        target = int(self.input_compression_threshold * self.input_compression_ratio)
        return compress_text(text, target)

    def _truncate_tool_content(self, content: str) -> str:
        if content is None:
            return None
        if len(content) <= self.tool_result_limit:
            return content
        self.last_report["tool_results_truncated"] += 1
        clipped = content[: self.tool_result_limit]
        # Cut at a word boundary and flag the truncation so the model
        # knows there is more (it can re-run a narrower query if needed)
        return clipped.rsplit(" ", 1)[0] + f"\n[... truncated: {len(content) - self.tool_result_limit} more chars ...]"

    def _digest(self, messages: List[Dict], char_budget: int) -> str:
        """One-paragraph summary of old turns used during compaction.

        Lines are individually short, and the total is trimmed (oldest
        first) to ``char_budget`` so the digest itself cannot blow the
        token budget it exists to enforce.
        """
        lines = []
        for message in messages:
            role = message.get("role", "?")
            content = message.get("content") or ""
            if role == "tool":
                name = message.get("name", "tool")
                first_line = content.split("\n", 1)[0]
                lines.append(f"- tool {name}: {first_line[:80]}")
            elif role == "assistant":
                if message.get("tool_calls"):
                    names = ", ".join(_tool_call_name(tc) for tc in message["tool_calls"])
                    lines.append(f"- assistant called tools: {names}")
                elif content:
                    lines.append(f"- assistant: {content[:80]}")
            elif role == "user" and content:
                lines.append(f"- user: {content[:80]}")

        header = "Summary of earlier conversation (auto-compacted to fit the token budget):"
        kept = lines
        omitted = 0
        while kept and len(header) + sum(len(l) + 1 for l in kept) > char_budget:
            kept = kept[1:]
            omitted += 1
        if omitted:
            kept.append(f"[... {omitted} older turn(s) omitted ...]")
        return header + "\n" + "\n".join(kept)

    def compact_history(self, messages: List[Dict]) -> List[Dict]:
        """Sticky compaction: rewrite the conversation in place when it
        exceeds the budget, returning the messages folded into the digest.

        View-only compaction (``prepare_messages``) rewrites the wire copy
        on every call once the history is over budget, so the prompt prefix
        changes each iteration and provider prompt caches can never hit.
        Rewriting in place - once, at the moment the budget is crossed -
        means subsequent iterations only append, and the prefix stays
        byte-stable between compaction events. Callers should persist the
        returned messages (the agent writes them to the transcript) so
        compaction never loses data.
        """
        if estimate_messages_tokens(messages) <= self.history_budget:
            return []

        # messages[0] = system, messages[1] = original user request (if present)
        head = messages[:2] if len(messages) > 1 and messages[1].get("role") == "user" else messages[:1]
        tail = messages[-self.recent_turns_kept :]
        middle = messages[len(head) : len(messages) - len(tail)]
        if not middle:
            return []

        # The digest gets whatever budget remains after the verbatim parts
        fixed_tokens = estimate_messages_tokens(head) + estimate_messages_tokens(tail)
        digest_budget = max(
            200,
            (self.history_budget - fixed_tokens) * CHARS_PER_TOKEN - len(middle) * 20,
        )
        digest = {"role": "user", "content": self._digest(middle, digest_budget)}
        dropped = [dict(message) for message in middle]
        messages[:] = head + [digest] + tail
        self.last_report["compacted"] = True
        return dropped

    def prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Apply truncation and compaction. Returns a new message list.

        The system prompt (first message) and the original user request are
        always preserved verbatim; only middle turns are ever compacted.
        """
        self.last_report = {"tool_results_truncated": 0, "compacted": False}
        if not messages:
            return messages

        prepared = []
        for message in messages:
            message = dict(message)
            if message.get("role") == "tool" and message.get("content"):
                message["content"] = self._truncate_tool_content(message["content"])
            prepared.append(message)

        if estimate_messages_tokens(prepared) <= self.history_budget:
            return prepared

        self.last_report["compacted"] = True
        # messages[0] = system, messages[1] = original user request (if present)
        head = prepared[:2] if len(prepared) > 1 and prepared[1].get("role") == "user" else prepared[:1]
        tail = prepared[-self.recent_turns_kept :]
        middle = prepared[len(head) : len(prepared) - len(tail)]
        if not middle:
            return prepared

        # The digest gets whatever budget remains after the verbatim parts
        fixed_tokens = estimate_messages_tokens(head) + estimate_messages_tokens(tail)
        digest_budget = max(
            200,
            (self.history_budget - fixed_tokens) * CHARS_PER_TOKEN - len(middle) * 20,
        )
        digest = {"role": "user", "content": self._digest(middle, digest_budget)}
        return head + [digest] + tail
