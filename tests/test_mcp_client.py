"""Tests for the MCP client over stdio (mcp_client.py).

These run a real subprocess speaking the protocol (a small Python script
written to the test tmp dir), so they exercise the actual JSON-RPC
transport, response correlation, and process lifecycle.
"""

import json
import sys

import pytest

from mcp_client import MCPClient, MCPToolWrapper, discover_mcp_tools

FAKE_SERVER = r'''
import json, sys

def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        return {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"}}
    if method == "tools/list":
        return {"tools": [{
            "name": "echo",
            "description": "Echo the text back",
            "inputSchema": {"type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"]},
        }]}
    if method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        if args.get("text") == "boom":
            return {"content": [{"type": "text", "text": "exploded"}], "isError": True}
        return {"content": [{"type": "text", "text": "echo: " + args.get("text", "")}]}
    return {}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}), flush=True)
'''


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


class TestMCPClient:
    def test_handshake_and_tool_listing(self, client):
        assert "echo" in client.tools
        assert client.tools["echo"]["description"] == "Echo the text back"

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
        bare = dict(client.tools["echo"])
        bare.pop("description")
        wrapper = MCPToolWrapper(client, bare)
        assert "echo" in wrapper.description
        assert "fake" in wrapper.description


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
