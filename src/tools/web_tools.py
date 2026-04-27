from langchain_tavily import TavilySearch
from src.config.settings import TAVILY_MAX_RESULTS

tavily_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)
