"""
web_tools.py — Web-search tool used by the Research Agent team.

This module wraps the Tavily Search API as a LangChain tool so that
LangGraph's ``create_react_agent`` can call it automatically when the
researcher agent needs external information.

Why Tavily?
  - Purpose-built for AI agents: returns clean, structured results instead
    of raw HTML.
  - Supports real-time web search without additional parsing overhead.

The ``TAVILY_API_KEY`` must be set in the .env file; the key is picked up
automatically by the ``langchain_tavily`` integration from the environment.
"""

from langchain_tavily import TavilySearch  # Official LangChain ↔ Tavily integration
from src.config.settings import TAVILY_MAX_RESULTS

# ── Tavily Search Tool ─────────────────────────────────────────────────────────
# max_results controls how many web results are returned per search query.
# Keeping this small (3) balances relevance with token usage — too many results
# can overflow the LLM's context window.
#
# This object is a LangChain @tool-compatible instance; LangGraph's ReAct agent
# will call it with a natural-language query and receive a list of results.
tavily_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)
