"""MCP (Model Context Protocol) client over stdio.

Tools so far come only from the local ``tools/`` directory. This module
lets the agent mount tools from any MCP server - filesystem access,
GitHub, databases, anything speaking the protocol - alongside them. The
stdio transport is implemented with the standard library only: a
subprocess, one JSON-RPC message per line, and a background reader
thread that correlates responses.

Config (all optional - no ``mcp`` section means zero MCP behavior):

    mcp:
      servers:
        - name: filesystem
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
          env: {}            # extra environment variables
          timeout: 30        # seconds per JSON-RPC round trip

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
from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "multi-agent-channel", "version": "2.0"}

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


class MCPClient:
    """One MCP server connection over stdio (JSON-RPC, newline-delimited)."""

    def __init__(self, name: str, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.timeout = float(timeout)
        self.proc: Optional[subprocess.Popen] = None
        self.tools: Dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._cond = threading.Condition()
        self._pending: Dict[int, Any] = {}  # request id -> result or MCPError
        self._dead = False
        self._reader: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        """Start the server, run the initialize handshake, and list tools."""
        try:
            self.proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env={**os.environ, **self.env},
            )
        except OSError as e:
            raise MCPError(f"could not start MCP server '{self.name}': {e}")

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self._notify("notifications/initialized")
        listed = self._request("tools/list", {})
        self.tools = {
            tool["name"]: tool
            for tool in (listed or {}).get("tools", [])
            if isinstance(tool, dict) and tool.get("name")
        }
        with _LIVE_LOCK:
            _LIVE_CLIENTS.append(self)

    def close(self) -> None:
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

        Notifications and server log lines (messages without an id) are
        ignored. On EOF every pending waiter is failed - a dead server
        must not hang the agent loop until its timeout.
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
                if not isinstance(message, dict) or "id" not in message:
                    continue
                if "error" in message:
                    entry = MCPError(str(message["error"].get("message") or "unknown MCP error"))
                else:
                    entry = message.get("result")
                with self._cond:
                    self._pending[message["id"]] = entry
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

    def _send(self, payload: dict) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' is not running")
        try:
            assert self.proc.stdin
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"lost MCP server '{self.name}': {e}")

    def _request(self, method: str, params: dict) -> Any:
        with self._cond:
            request_id = next(self._ids)
            self._send({"jsonrpc": "2.0", "id": request_id,
                        "method": method, "params": params})
            deadline = time.monotonic() + self.timeout
            while request_id not in self._pending:
                if self._dead:
                    raise MCPError(f"MCP server '{self.name}' is not running")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
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

    def call_tool(self, tool_name: str, arguments: Optional[dict]) -> dict:
        """Invoke a remote tool. Always returns a dict: {"text": ...} on
        success, {"error": ...} on failure - tool errors must surface to
        the model, not crash the agent loop."""
        try:
            result = self._request("tools/call", {"name": tool_name, "arguments": arguments or {}})
        except MCPError as e:
            return {"error": str(e)}

        content = result.get("content", []) if isinstance(result, dict) else []
        text = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
        if isinstance(result, dict) and result.get("isError"):
            return {"error": text or f"MCP tool '{tool_name}' failed"}
        return {"text": text or "(no output)"}


class MCPToolWrapper(BaseTool):
    """Adapts a remote MCP tool to the local BaseTool interface so it is
    discovered, schema'd, and callable exactly like a built-in tool."""

    def __init__(self, client: MCPClient, descriptor: dict):
        self._client = client
        self._descriptor = descriptor
        self._name = f"mcp_{_safe_name(client.name)}_{_safe_name(descriptor['name'])}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._descriptor.get("description") or (
            f"Tool '{self._descriptor['name']}' from MCP server '{self._client.name}'"
        )

    @property
    def parameters(self) -> dict:
        schema = dict(self._descriptor.get("inputSchema") or {})
        schema.setdefault("type", "object")
        return schema

    def execute(self, **kwargs):
        return self._client.call_tool(self._descriptor["name"], kwargs)


def discover_mcp_tools(config: dict, silent: bool = False):
    """Connect to the configured MCP servers and return their tools.

    Returns ``(tools, clients)``. A server that fails to start contributes
    no tools and only a warning - MCP is strictly additive.
    """
    tools: Dict[str, BaseTool] = {}
    clients: List[MCPClient] = []
    servers = (config.get("mcp") or {}).get("servers") or []
    for server in servers:
        name = server.get("name") or "mcp"
        try:
            client = MCPClient(
                name=name,
                command=server["command"],
                args=server.get("args") or [],
                env=server.get("env") or {},
                timeout=server.get("timeout", 30),
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
