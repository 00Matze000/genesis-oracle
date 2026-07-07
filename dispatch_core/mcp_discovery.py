"""MCP-Tool-Discovery-Demo (Problem Set 12, Submission: Python Source).

Zeigt, dass der SpreelandDispatcher seine Infrastruktur-Tools tatsaechlich zur
Laufzeit ueber das Model Context Protocol *entdeckt* -- ohne LLM-Aufruf, damit
der Nachweis reproduzierbar und quota-frei ist. Verwendet dieselbe
ADK-``McpToolset``, die auch der Agent nutzt, verbindet sich per stdio mit dem
Bridge-MCP-Server und ruft ``get_tools()`` auf.

Lauf:  uv run --native-tls python -m dispatch_core.mcp_discovery
"""

import asyncio
import sys
from pathlib import Path

from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

_HERE = Path(__file__).resolve().parent
_BRIDGE_SERVER = _HERE / "bridge_mcp_server.py"


async def discover() -> None:
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[str(_BRIDGE_SERVER)],
            )
        )
    )
    try:
        tools = await toolset.get_tools()
        print(f"MCP-Server verbunden. {len(tools)} Tool(s) entdeckt:\n")
        for t in tools:
            schema = getattr(t.raw_mcp_tool, "inputSchema", None) or {}
            props = schema.get("properties", {}) or {}
            params = ", ".join(props.keys())
            print(f"  - {t.name}({params})")
            print(f"      {t.description.strip().splitlines()[0]}")

        # Round-Trip-Beweis: ein entdecktes Tool real aufrufen.
        status_tool = next(t for t in tools if t.name == "get_bridge_status")
        result = await status_tool.run_async(
            args={"bridge_id": "burg-01"}, tool_context=None
        )
        print("\nAufruf get_bridge_status('burg-01') ->")
        print(f"  {result}")
    finally:
        await toolset.close()


if __name__ == "__main__":
    asyncio.run(discover())
