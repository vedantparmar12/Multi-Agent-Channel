"""MCP (Model Context Protocol) client over stdio.

Tools so far come only from the local ``tools/`` directory. This module
lets the agent mount tools from any MCP server - filesystem access,
GitHub, databases, anything speaking the protocol - alongside them. The
stdio transport is implemented with the standard library only: a
subprocess, one JSON-RPC message per line, and a background reader
thread that correlates responses.

Implements the client half of the MCP specification and its documented
best practices: protocol-version negotiation on initialize (an
unsupported version means a clean disconnect), paginated ``tools/list``
via ``nextCursor``, live tool-registry reload on
``notifications/tools/list_changed`` (tool wrappers read their
descriptor live so schema changes take effect), JSON-RPC replies to
server-initiated requests (``ping`` is answered, anything unexpected
gets ``-32601``), a cancellation notification when a request times out,
server stderr captured to ``logs/mcp-<server>.log`` (never mixed into
the protocol stream), and the specified stdio shutdown order: close
stdin, wait for the server to exit, then SIGTERM, then SIGKILL.

Config (all optional - no ``mcp`` section means zero MCP behavior); both
the mapping style used by most MCP hosts and a plain list work:

    mcp:
      servers:
        filesystem:               # name -> config (mapping style)
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
          env: {}                  # extra environment variables
          timeout: 30              # seconds per JSON-RPC round trip

Failures degrade to a warning: a server that cannot start or dies
mid-session simply stops contributing tools, and live subprocesses are
terminated at interpreter exit.
"""

import atexit
import itertools
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool

PROTOCOL_VERSION = "2025-06-18"
# Versions whose initialize / tools/list / tools/call wire behavior this
# client speaks; a server answering with anything else is disconnected.
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05", "2025-03-26", "2025-06-18", "2026-07-28",
}
CLIENT_INFO = {"name": "multi-agent-channel", "version": "2.1"}

# Pagination guard: a misbehaving server cannot loop tools/list forever
MAX_LIST_PAGES = 100
# Seconds a server gets to exit on its own after stdin closes (the spec's
# shutdown order) before SIGTERM/SIGKILL escalation
EXIT_GRACE_SECONDS = 2.0

# Live clients, closed at interpreter exit so MCP subprocesses never linger
_LIVE_CLIENTS: List["MCPClient"] = []
_LIVE_LOCK = threading.Lock()


@atexit.register
def _close_live_clients() -> None:
    for client in list(_LIVE_CLIENTS):
        try:
            client.close()
        except Exception:
            pass


class MCPError(Exception):
    """Transport- or protocol-level failure talking to an MCP server."""


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "tool"


def _content_text(part: dict) -> str:
    """Render one tools/call content item as model-safe text.

    Binary content (image/audio) is never inlined - the agent loop wants
    text, and base64 blobs would only burn tokens.
    """
    kind = part.get("type")
    if kind == "text":
        return str(part.get("text", ""))
    if kind == "image":
        return f"[image: {part.get('mimeType', 'unknown')} omitted]"
    if kind == "audio":
        return f"[audio: {part.get('mimeType', 'unknown')} omitted]"
    if kind == "resource_link":
        return f"[resource: {part.get('uri', '')}]"
    if kind == "embedded_resource":
        resource = part.get("resource") or {}
        if isinstance(resource.get("text"), str):
            return f"[resource {resource.get('uri', '')}]\n{resource['text']}"
        return f"[resource: {resource.get('uri', 'unknown')}]"
    return f"[{kind or 'unknown'} content omitted]"


class MCPClient:
    """One MCP server connection over stdio (JSON-RPC, newline-delimited)."""

    def __init__(self, name: str, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, timeout: float = 30.0,
                 silent: bool = False):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.timeout = float(timeout)
        self.silent = silent
        self.proc: Optional[subprocess.Popen] = None
        self.tools: Dict[str, dict] = {}
        # What the server actually answered with, after negotiation
        self.protocol_version: Optional[str] = None
        self.server_info: Dict[str, Any] = {}
        self.server_capabilities: Dict[str, Any] = {}
        self._ids = itertools.count(1)
        self._cond = threading.Condition()
        self._pending: Dict[Any, Any] = {}  # request id -> result or MCPError
        self._awaiting: set = set()  # request ids still expecting a reply
        self._send_lock = threading.Lock()
        self._dead = False
        self._reader: Optional[threading.Thread] = None
        self._stderr_path = Path("logs") / f"mcp-{_safe_name(name)}.log"

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        """Start the server, negotiate the protocol version, list tools."""
        try:
            self.proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env={**os.environ, **self.env},
            )
        except OSError as e:
            raise MCPError(f"could not start MCP server '{self.name}': {e}")

        # Registered immediately: a failed handshake must not leak the
        # subprocess (close() is also called on that path below)
        with _LIVE_LOCK:
            _LIVE_CLIENTS.append(self)

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        stderr_thread.start()

        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            })
            if not isinstance(result, dict):
                raise MCPError(
                    f"MCP server '{self.name}' returned an invalid initialize response"
                )
            version = result.get("protocolVersion")
            if version not in SUPPORTED_PROTOCOL_VERSIONS:
                # Spec: a version the client cannot speak means disconnect
                raise MCPError(
                    f"MCP server '{self.name}' speaks unsupported protocol "
                    f"version {version!r}"
                )
            self.protocol_version = version
            self.server_info = result.get("serverInfo") or {}
            self.server_capabilities = result.get("capabilities") or {}
            self._notify("notifications/initialized")
            self.tools = self._list_tools()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Shut the server down in the spec's order: close stdin, give it
        a moment to exit on EOF, then SIGTERM, then SIGKILL."""
        proc, self.proc = self.proc, None
        with _LIVE_LOCK:
            if self in _LIVE_CLIENTS:
                _LIVE_CLIENTS.remove(self)
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=EXIT_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # -- transport -----------------------------------------------------

    def _read_loop(self) -> None:
        """Background reader: match responses to pending requests.

        Server-initiated requests are answered per JSON-RPC (ping gets an
        empty result, anything unexpected gets -32601 - otherwise a
        pinging server blocks forever), ``notifications/tools/list_changed``
        triggers a registry reload, and on EOF every pending waiter is
        failed - a dead server must not hang the agent loop until its
        timeout.
        """
        try:
            assert self.proc and self.proc.stdout
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue

                if "method" in message and "id" in message:
                    try:
                        self._handle_server_request(message)
                    except Exception:
                        pass
                    continue
                if "method" in message:
                    self._handle_notification(message)
                    continue

                request_id = message.get("id")
                if "error" in message:
                    entry = MCPError(str(message["error"].get("message") or "unknown MCP error"))
                else:
                    entry = message.get("result")
                with self._cond:
                    # Late replies to timed-out/cancelled requests are dropped
                    if request_id in self._awaiting:
                        self._awaiting.discard(request_id)
                        self._pending[request_id] = entry
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._dead = True
                for request_id in self._pending:
                    if not isinstance(self._pending[request_id], Exception):
                        self._pending[request_id] = MCPError(
                            f"MCP server '{self.name}' closed the connection"
                        )
                self._cond.notify_all()

    def _handle_server_request(self, message: dict) -> None:
        """Reply to a server-initiated request. We declare no optional
        client capabilities, so ping is the only one a compliant server
        sends - anything else is politely refused with -32601."""
        method = message.get("method")
        request_id = message.get("id")
        if method == "ping":
            self._try_send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        else:
            self._try_send({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })

    def _handle_notification(self, message: dict) -> None:
        if message.get("method") == "notifications/tools/list_changed":
            # Refresh off the reader thread: the refresh itself waits for
            # a reply only this thread can deliver
            threading.Thread(target=self._refresh_tools, daemon=True).start()

    def _refresh_tools(self) -> None:
        """Re-run tools/list after the server announced a change."""
        try:
            tools = self._list_tools()
        except Exception:
            return
        with self._cond:
            if self._dead:
                return
            self.tools = tools
        if not self.silent:
            print(f"🔌 MCP server '{self.name}' tool list changed: {len(tools)} tool(s)")

    def _stderr_loop(self) -> None:
        """Capture server stderr (its logs, per the spec) to a file.

        Best effort: capture failures are ignored, and the pipe is always
        drained so a chatty server never blocks on a full buffer.
        """
        try:
            assert self.proc and self.proc.stderr
            self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
            with self._stderr_path.open("a", encoding="utf-8", errors="replace") as log:
                for line in self.proc.stderr:
                    log.write(line if line.endswith("\n") else line + "\n")
                    log.flush()
        except Exception:
            pass

    def _send(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' is not running")
        try:
            assert proc.stdin
            # The reader thread and refresh threads also send; one line at
            # a time under the lock keeps JSON messages from interleaving
            with self._send_lock:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"lost MCP server '{self.name}': {e}")

    def _try_send(self, payload: dict) -> None:
        try:
            self._send(payload)
        except Exception:
            pass

    def _request(self, method: str, params: dict) -> Any:
        with self._cond:
            request_id = next(self._ids)
            self._awaiting.add(request_id)
            try:
                self._send({"jsonrpc": "2.0", "id": request_id,
                            "method": method, "params": params})
            except Exception:
                self._awaiting.discard(request_id)
                raise
            deadline = time.monotonic() + self.timeout
            while request_id not in self._pending:
                if self._dead:
                    self._awaiting.discard(request_id)
                    raise MCPError(f"MCP server '{self.name}' is not running")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._awaiting.discard(request_id)
                    # Tell the server to stop working on it (best effort;
                    # a late reply is dropped by the reader)
                    self._try_send({
                        "jsonrpc": "2.0", "method": "notifications/cancelled",
                        "params": {"requestId": request_id},
                    })
                    raise MCPError(
                        f"timed out waiting for '{method}' from MCP server '{self.name}'"
                    )
                self._cond.wait(remaining)
            entry = self._pending.pop(request_id)
        if isinstance(entry, Exception):
            raise entry
        return entry

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    # -- tools -----------------------------------------------------------

    def _list_tools(self) -> Dict[str, dict]:
        """Fetch the tool list, following nextCursor pagination."""
        collected: Dict[str, dict] = {}
        cursor = None
        seen_cursors = set()
        for _ in range(MAX_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            listed = self._request("tools/list", params)
            if not isinstance(listed, dict):
                break
            for tool in listed.get("tools", []):
                if isinstance(tool, dict) and tool.get("name"):
                    collected[tool["name"]] = tool
            cursor = listed.get("nextCursor")
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        return collected

    def call_tool(self, tool_name: str, arguments: Optional[dict]) -> dict:
        """Invoke a remote tool. Always returns a dict: {"text": ...} on
        success, {"error": ...} on failure - tool errors must surface to
        the model, not crash the agent loop."""
        try:
            result = self._request("tools/call", {"name": tool_name, "arguments": arguments or {}})
        except MCPError as e:
            return {"error": str(e)}

        if not isinstance(result, dict):
            return {"text": "(no output)"}
        parts = [
            _content_text(part)
            for part in result.get("content", [])
            if isinstance(part, dict)
        ]
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            parts.append("structured: " + json.dumps(structured, ensure_ascii=False))
        text = "\n".join(part for part in parts if part).strip()
        if result.get("isError"):
            return {"error": text or f"MCP tool '{tool_name}' failed"}
        return {"text": text or "(no output)"}


class MCPToolWrapper(BaseTool):
    """Adapts a remote MCP tool to the local BaseTool interface so it is
    discovered, schema'd, and callable exactly like a built-in tool."""

    def __init__(self, client: MCPClient, descriptor: dict):
        self._client = client
        self._remote_name = descriptor["name"]
        self._fallback = descriptor
        self._name = f"mcp_{_safe_name(client.name)}_{_safe_name(self._remote_name)}"

    def _descriptor(self) -> dict:
        # Live lookup so notifications/tools/list_changed refreshes flow
        # through to schema and description; the discovery-time copy is
        # the fallback if the server dropped the tool mid-session.
        return self._client.tools.get(self._remote_name) or self._fallback

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        descriptor = self._descriptor()
        return descriptor.get("description") or (
            f"Tool '{self._remote_name}' from MCP server '{self._client.name}'"
        )

    @property
    def parameters(self) -> dict:
        schema = dict(self._descriptor().get("inputSchema") or {})
        schema.setdefault("type", "object")
        return schema

    def execute(self, **kwargs):
        return self._client.call_tool(self._remote_name, kwargs)


def _iter_servers(servers):
    """Yield (name, config) pairs from either servers config style:
    a name->config mapping (the convention most MCP hosts use) or a
    plain list of configs that each carry their own name."""
    if isinstance(servers, dict):
        for name, config in servers.items():
            yield str(name), config if isinstance(config, dict) else {}
    elif isinstance(servers, list):
        for config in servers:
            if isinstance(config, dict):
                yield config.get("name") or "mcp", config


def discover_mcp_tools(config: dict, silent: bool = False):
    """Connect to the configured MCP servers and return their tools.

    Returns ``(tools, clients)``. A server that fails to start contributes
    no tools and only a warning - MCP is strictly additive.
    """
    tools: Dict[str, BaseTool] = {}
    clients: List[MCPClient] = []
    servers = (config.get("mcp") or {}).get("servers") or []
    for name, server in _iter_servers(servers):
        try:
            client = MCPClient(
                name=name,
                command=server["command"],
                args=server.get("args") or [],
                env=server.get("env") or {},
                timeout=server.get("timeout", 30),
                silent=silent,
            )
            client.connect()
        except (KeyError, TypeError, MCPError) as e:
            if not silent:
                print(f"⚠️  MCP server '{name}' unavailable: {e}")
            continue
        clients.append(client)
        for descriptor in client.tools.values():
            wrapper = MCPToolWrapper(client, descriptor)
            tools[wrapper.name] = wrapper
        if not silent:
            print(f"🔌 MCP server '{name}': {len(client.tools)} tool(s)")
    return tools, clients
