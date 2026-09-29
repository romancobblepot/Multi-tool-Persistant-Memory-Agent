import os
import requests
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from runtime import run_async

client = MultiServerMCPClient({
    "expense": {"transport": "streamable_http", "url": "https://splendid-gold-dingo.fastmcp.app/mcp"},
    "manim-server": {
        "transport": "stdio",
        "command": "/Users/adnaniqbalkantroo/Langgraph-Agents-tutorial/.venv/bin/python",
        "args": ["/Users/adnaniqbalkantroo/Langgraph-Agents-tutorial/manim-mcp-server/src/manim_server.py"],
        "env": {"MANIM_EXECUTABLE": "/Users/adnaniqbalkantroo/Langgraph-Agents-tutorial/.venv/bin/manim"},
    },
    "tavily-remote-mcp": {
        "transport": "streamable_http",
        "url": "https://mcp.tavily.com/mcp/" f"?tavilyApiKey={os.environ['TAVILY_API_KEY']}",
    },
})


def load_mcp_tools() -> list[BaseTool]:
    return run_async(client.get_tools())


mcp_tools = load_mcp_tools()
tavily_search_tool = next((tool for tool in mcp_tools if tool.name == "tavily_search"), None)
if tavily_search_tool is None:
    raise RuntimeError(f"Tavily search tool was not loaded. Available MCP tools: {[tool.name for tool in mcp_tools]}")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest Alpha Vantage quote for a stock symbol such as AAPL."""
    response = requests.get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": symbol.strip("() '\"").upper(), "apikey": os.getenv("STOCK_API_KEY")}, timeout=20)
    response.raise_for_status()
    return response.json()
