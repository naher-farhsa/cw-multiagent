from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from src.agent.llm import llm
from src.tools.web_tools import tavily_tool

researcher_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=(
        "You are a research assistant. Use the Tavily search tool to gather "
        "accurate, up-to-date information on the user's topic. Summarize key "
        "findings as concise bullet points with source URLs. Do not ask follow-up questions."
    ),
)


def researcher_node(state) -> Command[Literal["supervisor"]]:
    result = researcher_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )
