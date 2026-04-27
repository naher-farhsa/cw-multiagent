from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from src.agent.llm import llm
from src.tools.file_tools import write_document, read_document

writer_agent = create_react_agent(
    llm,
    tools=[write_document, read_document],
    prompt=(
        "You are a content writer. Using the research provided in the conversation, "
        "write a well-structured article in Markdown with a clear title, introduction, "
        "sections with headings, and a conclusion. Save the final article using the "
        "'write_document' tool with a descriptive .md filename. Do not ask follow-up questions."
    ),
)


def writer_node(state) -> Command[Literal["supervisor"]]:
    result = writer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="writer")
            ]
        },
        goto="supervisor",
    )
