"""Tests for the MCP client over stdio (mcp_client.py).

These run a real subprocess speaking the protocol (a small Python script
written to the test tmp dir), so they exercise the actual JSON-RPC
transport, response correlation, and process lifecycle.

The fake server is configurable through environment variables so one
script covers every scenario: protocol version, tools/list pagination,
notifications/tools/list_changed, server-initiated ping/requests, stderr
logging, hung tool calls (for cancellation), and rich content types.
"""

import json
import sys
import time

import pytest

from mcp_client import MCPClient, MCPToolWrapper, discover_mcp_tools

FAKE_SERVER = r'''
import json, os, sys

ENV = os.environ
STATE = {"mutated": False, "notified": False}

def tools_now():
    tools = [{
        "name": "echo",
        "description": "Echo the text back (v2)" if STATE["mutated"] else "Echo the text back",
        "inputSchema": {"type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
    }]
    if STATE["mutated"]:
        tools.append({"name": "spawned", "description": "Added after list_changed",
                      "inputSchema": {"type": "object", "properties": {}}})
    return tools

def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        return {"protocolVersion": ENV.get("FAKE_PROTOCOL_VERSION", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"}}
    if method == "tools/list":
        if ENV.get("FAKE_PAGINATE"):
            cursor = (msg.get("params") or {}).get("cursor")
            if cursor is None:
                return {"tools": tools_now(), "nextCursor": "page2"}
            return {"tools": [{"name": "late", "description": "Page two",
                               "inputSchema": {"type": "object", "properties": {}}}]}
        return {"tools": tools_now()}
    if method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        if args.get("text") == "boom":
            return {"content": [{"type": "text", "text": "exploded"}], "isError": True}
        if args.get("text") == "mutate":
            return {"content": [{"type": "text", "text": "mutated"}]}
        if ENV.get("FAKE_RICH_CONTENT"):
            return {"content": [
                        {"type": "text", "text": "caption"},
                        {"type": "image", "mimeType": "image/png", "data": "iVBOR"},
                        {"type": "resource_link", "uri": "file:///tmp/x.txt", "name": "x"},
                    ],
                    "structuredContent": {"answer": 42}}
        return {"content": [{"type": "text", "text": "echo: " + args.get("text", "")}]}
    return {}

def send(obj):
    print(json.dumps(obj), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")

    if method == "notifications/initialized":
        if ENV.get("FAKE_PING"):
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
        if ENV.get("FAKE_UNKNOWN_REQUEST"):
            send({"jsonrpc": "2.0", "id": "srv-2", "method": "roots/list"})
        if ENV.get("FAKE_STDERR_LINE"):
            print(ENV["FAKE_STDERR_LINE"], file=sys.stderr, flush=True)
        continue
    if method == "notifications/cancelled":
        if ENV.get("FAKE_CANCEL_LOG"):
            with open(ENV["FAKE_CANCEL_LOG"], "a") as f:
                f.write(json.dumps(msg) + "\n")
        continue
    if "method" not in msg:
        # Response to a server-initiated request - record for assertions
        if "id" in msg and ENV.get("FAKE_MARKER"):
            with open(ENV["FAKE_MARKER"], "a") as f:
                f.write(json.dumps(msg) + "\n")
        continue
    if "id" not in msg:
        continue
    if ENV.get("FAKE_HANG") and method == "tools/call":
        continue  # never answer: forces the client timeout + cancellation
    send({"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)})
    if method == "tools/call" and \
            (msg.get("params", {}).get("arguments") or {}).get("text") == "mutate":
        STATE["mutated"] = True
        if not STATE["notified"]:
            STATE["notified"] = True
            send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
'''


@pytest.fixture(autouse=True)
def _chdir(tmp_path, monkeypatch):
    # stderr logs (logs/mcp-<name>.log) land in the test tmp dir, not the repo
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return str(path)


@pytest.fixture
def client(server_path):
    connected = MCPClient("fake", sys.executable, [server_path])
    connected.connect()
    yield connected
    connected.close()


def make_client(tmp_path, env=None, name="fake", timeout=30.0):
    path = tmp_path / f"server_{name}.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return MCPClient(name, sys.executable, [str(path)], env=env or {}, timeout=timeout)


def wait_until(condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


class TestMCPClient:
    def test_handshake_and_tool_listing(self, client):
        assert "echo" in client.tools
        assert client.tools["echo"]["description"] == "Echo the text back"
        assert client.protocol_version == "2025-06-18"
        assert client.server_info["name"] == "fake"

    def test_call_tool_round_trip(self, client):
        result = client.call_tool("echo", {"text": "hello"})
        assert result == {"text": "echo: hello"}

    def test_tool_error_surfaces_as_dict(self, client):
        result = client.call_tool("echo", {"text": "boom"})
        assert "error" in result
        assert "exploded" in result["error"]

    def test_dead_server_fails_instead_of_hanging(self, server_path):
        dying = MCPClient("dying", sys.executable, ["-c", "import sys; sys.exit(1)"])
        with pytest.raises(Exception):
            dying.connect()

    def test_close_terminates_process(self, server_path):
        one = MCPClient("fake", sys.executable, [server_path])
        one.connect()
        proc = one.proc
        one.close()
        assert one.proc is None
        proc.wait(timeout=5)
        assert proc.poll() is not None

    def test_timeout_raises(self, server_path):
        slow = MCPClient(
            "slow", sys.executable,
            ["-c", "import sys, time, json\n"
             "for line in sys.stdin:\n"
             "    time.sleep(5)\n"],
            timeout=0.5,
        )
        with pytest.raises(Exception, match="timed out"):
            slow.connect()
        slow.close()


class TestBestPractices:
    """Behaviors required by the MCP specification and client best
    practices docs, exercised against the configurable fake server."""

    def test_version_negotiation_accepts_known_versions(self, tmp_path):
        older = make_client(tmp_path, env={"FAKE_PROTOCOL_VERSION": "2025-03-26"})
        older.connect()
        try:
            assert older.protocol_version == "2025-03-26"
            assert "echo" in older.tools
        finally:
            older.close()

    def test_unsupported_version_disconnects_without_leaking(self, tmp_path):
        rogue = make_client(tmp_path, env={"FAKE_PROTOCOL_VERSION": "1999-01-01"})
        with pytest.raises(Exception, match="protocol"):
            rogue.connect()
        assert rogue.proc is None  # failed handshake cleaned the subprocess up

    def test_paginated_tool_listing(self, tmp_path):
        paged = make_client(tmp_path, env={"FAKE_PAGINATE": "1"})
        paged.connect()
        try:
            assert {"echo", "late"} <= set(paged.tools)
        finally:
            paged.close()

    def test_ping_answered_and_unknown_request_refused(self, tmp_path):
        marker = tmp_path / "marker.jsonl"
        chatty = make_client(tmp_path, env={
            "FAKE_PING": "1",
            "FAKE_UNKNOWN_REQUEST": "1",
            "FAKE_MARKER": str(marker),
        })
        chatty.connect()
        try:
            assert wait_until(lambda: marker.exists())
            entries = [json.loads(line) for line in
                       marker.read_text().splitlines() if line.strip()]
            by_id = {entry.get("id"): entry for entry in entries}
            assert by_id.get("srv-1", {}).get("result") == {}
            assert by_id.get("srv-2", {}).get("error", {}).get("code") == -32601
        finally:
            chatty.close()

    def test_list_changed_refreshes_registry(self, tmp_path):
        mutable = make_client(tmp_path)
        mutable.connect()
        wrapper = MCPToolWrapper(mutable, mutable.tools["echo"])
        try:
            assert "spawned" not in mutable.tools
            assert mutable.call_tool("echo", {"text": "mutate"}) == {"text": "mutated"}

            assert wait_until(lambda: "spawned" in mutable.tools)
            # Wrappers read descriptors live, so the updated description
            # flows through without rebuilding anything
            assert wrapper.description == "Echo the text back (v2)"
            assert mutable.call_tool("spawned", {}) == {"text": "echo:"}
        finally:
            mutable.close()

    def test_stderr_is_captured_to_log(self, tmp_path):
        loud = make_client(tmp_path, env={"FAKE_STDERR_LINE": "server starting up"})
        loud.connect()
        try:
            log = tmp_path / "logs" / "mcp-fake.log"
            assert wait_until(lambda: log.exists())
            assert "server starting up" in log.read_text(encoding="utf-8")
        finally:
            loud.close()

    def test_timeout_sends_cancellation_notification(self, tmp_path):
        cancel_log = tmp_path / "cancelled.jsonl"
        hanging = make_client(
            tmp_path,
            env={"FAKE_HANG": "1", "FAKE_CANCEL_LOG": str(cancel_log)},
            timeout=0.5,
        )
        hanging.connect()
        try:
            result = hanging.call_tool("echo", {"text": "hang"})
            assert "error" in result
            assert "timed out" in result["error"]
            assert wait_until(lambda: cancel_log.exists())
            notifications = [json.loads(line) for line in
                             cancel_log.read_text().splitlines() if line.strip()]
            assert any(
                note.get("method") == "notifications/cancelled" and
                note.get("params", {}).get("requestId") is not None
                for note in notifications
            )
        finally:
            hanging.close()

    def test_rich_content_types_and_structured_output(self, tmp_path):
        rich = make_client(tmp_path, env={"FAKE_RICH_CONTENT": "1"})
        rich.connect()
        try:
            result = rich.call_tool("echo", {"text": "anything"})
            text = result["text"]
            assert "caption" in text
            assert "image/png" in text
            assert "file:///tmp/x.txt" in text
            assert "structured" in text and '"answer": 42' in text
        finally:
            rich.close()

    def test_shutdown_is_graceful_spec_order(self, tmp_path):
        polite = make_client(tmp_path)
        polite.connect()
        proc = polite.proc
        polite.close()
        # stdin closed -> server exits on EOF by itself: returncode 0,
        # no SIGTERM/SIGKILL escalation
        assert proc.wait(timeout=5) == 0


class TestMCPToolWrapper:
    def test_adapts_descriptor_to_local_tool(self, client):
        wrapper = MCPToolWrapper(client, client.tools["echo"])
        assert wrapper.name == "mcp_fake_echo"
        assert wrapper.parameters["properties"]["text"]["type"] == "string"
        schema = wrapper.to_openrouter_schema()
        assert schema["function"]["name"] == "mcp_fake_echo"
        assert schema["type"] == "function"
        assert wrapper.execute(text="round trip") == {"text": "echo: round trip"}

    def test_missing_description_gets_default(self, client):
        # A tool the registry doesn't know exercises the fallback path
        ghost = {"name": "ghost", "inputSchema": {"type": "object"}}
        wrapper = MCPToolWrapper(client, ghost)
        assert "ghost" in wrapper.description
        assert "fake" in wrapper.description
        assert wrapper.parameters["type"] == "object"


class TestDiscovery:
    def test_discovers_configured_servers(self, server_path):
        config = {"mcp": {"servers": [
            {"name": "fake", "command": sys.executable, "args": [server_path]},
        ]}}
        tools, clients = discover_mcp_tools(config, silent=True)
        try:
            assert "mcp_fake_echo" in tools
            assert len(clients) == 1
            assert tools["mcp_fake_echo"].execute(text="via discovery") == {
                "text": "echo: via discovery"
            }
        finally:
            for c in clients:
                c.close()

    def test_discovers_mapping_style_servers(self, server_path):
        # The config.yaml.example style: name -> config mapping
        config = {"mcp": {"servers": {
            "fake": {"command": sys.executable, "args": [server_path]},
        }}}
        tools, clients = discover_mcp_tools(config, silent=True)
        try:
            assert "mcp_fake_echo" in tools
            assert tools["mcp_fake_echo"].execute(text="mapped") == {
                "text": "echo: mapped"
            }
        finally:
            for c in clients:
                c.close()

    def test_unstartable_server_contributes_nothing(self):
        config = {"mcp": {"servers": [
            {"name": "broken", "command": sys.executable,
             "args": ["-c", "raise SystemExit(1)"]},
        ]}}
        tools, clients = discover_mcp_tools(config, silent=True)
        assert tools == {}
        assert clients == []

    def test_no_mcp_section_is_noop(self):
        tools, clients = discover_mcp_tools({}, silent=True)
        assert tools == {}
        assert clients == []


class TestAgentIntegration:
    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch, server_path):
        monkeypatch.chdir(tmp_path)
        exe = sys.executable.replace("\\", "/")
        server = server_path.replace("\\", "/")
        (tmp_path / "config.yaml").write_text(
            f"""
openrouter:
  base_url: "https://openrouter.ai/api/v1"
  api_key: "test-key"
  model: "test-model"
system_prompt: "You are a test agent."
agent:
  max_iterations: 2
memory:
  enabled: false
mcp:
  servers:
    - name: fake
      command: "{exe}"
      args: ["{server}"]
""",
            encoding="utf-8",
        )
        return tmp_path

    def test_agent_mounts_mcp_tools(self, workdir):
        from agent import OpenRouterAgent
        agent = OpenRouterAgent(silent=True)
        try:
            assert "mcp_fake_echo" in agent.tool_mapping
            assert any(
                tool["function"]["name"] == "mcp_fake_echo" for tool in agent.tools
            )
            result = agent.tool_mapping["mcp_fake_echo"](text="mounted")
            assert result == {"text": "echo: mounted"}
        finally:
            agent.close()

    def test_close_releases_mcp_processes(self, workdir):
        from agent import OpenRouterAgent
        agent = OpenRouterAgent(silent=True)
        procs = [c.proc for c in agent.mcp_clients]
        agent.close()
        assert agent.mcp_clients == []
        for proc in procs:
            proc.wait(timeout=5)
            assert proc.poll() is not None
