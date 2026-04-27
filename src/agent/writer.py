"""
writer.py — Writing Agent Team (writer, note-taker, chart-generator roles).

Architecture (from the workflow diagram):
  Supervisor Agent → Writing Agent Team → Supervisor Agent

The Writing Agent Team maps to three distinct roles, all handled by this single
ReAct agent with three tools:

  ┌─────────────────┬──────────────────────┬──────────────────────────────────┐
  │ Diagram Role    │ Tool                 │ What it does                     │
  ├─────────────────┼──────────────────────┼──────────────────────────────────┤
  │ writer          │ write_document       │ Drafts & saves the Markdown file │
  │ note-taker      │ read_document        │ Re-reads saved drafts for review │
  │ chart-generator │ generate_chart       │ Creates a matplotlib .png chart  │
  └─────────────────┴──────────────────────┴──────────────────────────────────┘

Typical execution flow:
  1. The agent receives the researcher's bullet-point findings in the messages.
  2. It calls ``generate_chart`` if the data warrants a visual (e.g. stats).
  3. It calls ``write_document`` to save the complete Markdown article
     (embedding the chart image path with ![caption](path)).
  4. It optionally calls ``read_document`` to verify the saved content.
  5. It returns a final confirmation message; the supervisor sees this and
     decides to FINISH.
"""

from typing import Literal
from langchain_core.messages import AIMessage       # wraps agent output as an AI message
from langgraph.prebuilt import create_react_agent   # prebuilt ReAct agent sub-graph factory
from langgraph.types import Command                 # routing instruction for the parent graph

from src.agent.llm import llm                                          # shared Groq LLM instance
from src.tools.file_tools import write_document, read_document, generate_chart  # all three tools

# ── Writer ReAct Agent ─────────────────────────────────────────────────────────
# The agent is given all three Writing Team tools so it can autonomously decide
# whether to generate a chart before writing the article, or skip it if the
# topic is not data-heavy.
#
# System prompt tells the agent:
#   - Its three roles (writer, note-taker, chart-generator).
#   - When to use each tool.
#   - That it must save the article with write_document before finishing.
#   - That it must NOT ask follow-up questions — act autonomously.
writer_agent = create_react_agent(
    llm,
    tools=[write_document, read_document, generate_chart],  # all Writing Team tools
    prompt=(
        "You are a content writing team with three roles:\n"
        "  1. WRITER: Write a well-structured Markdown article with a clear title, "
        "introduction, sections with headings, and a conclusion based on the research provided.\n"
        "  2. NOTE-TAKER: Use 'read_document' if you need to review a previously saved draft.\n"
        "  3. CHART-GENERATOR: If the research contains numerical data or statistics "
        "(e.g. market share percentages, growth rates, year-over-year comparisons), "
        "use 'generate_chart' to create a visual chart BEFORE writing the article, "
        "then embed the chart in the article using Markdown image syntax: "
        "![Chart Title](full/path/to/chart.png).\n\n"
        "Finally, save the complete article using 'write_document' with a descriptive .md filename. "
        "Do not ask follow-up questions."
    ),
)


# ── Writer Node ────────────────────────────────────────────────────────────────
def writer_node(state) -> Command[Literal["supervisor"]]:
    """Execute the Writing Agent Team and route the result back to the supervisor.

    This function is registered as the ``"writer"`` node in the LangGraph
    ``StateGraph``.  It is called by LangGraph whenever the supervisor decides
    the next step is ``"writer"``.

    Workflow:
        1. Pass the current shared ``state`` (messages including researcher
           findings) to the writer ReAct agent.
        2. The agent autonomously decides which tools to call and in what order:
           - ``generate_chart``  (optional — only if data warrants a visual)
           - ``write_document``  (mandatory — saves the finished article)
           - ``read_document``   (optional — verifies the saved content)
        3. The final text response (save confirmation) is extracted from the
           last message in the agent's output.
        4. A ``HumanMessage`` tagged with ``name="writer"`` is returned so the
           supervisor can identify the message origin when making its next
           routing decision (which will typically be FINISH).
        5. A ``Command`` routes control back to the supervisor.

    Args:
        state: The current ``MessagesState`` dict shared across all graph nodes,
               containing at minimum the user request and researcher output.

    Returns:
        A ``Command`` that appends the writer's final message to the shared
        state and routes to ``"supervisor"``.

    Note:
        The supervisor is expected to return ``FINISH`` after seeing a message
        like "Document saved to 'doc-result/....md'." from the writer.
    """
    # Run the writer ReAct sub-graph.  It will autonomously:
    #   - decide whether to call generate_chart first
    #   - call write_document to persist the article
    # create_react_agent handles the tool-call loop internally.
    result = writer_agent.invoke(state)

    return Command(
        update={
            # Expose only the final assistant message to the parent graph.
            # The supervisor reads this to decide whether the task is complete.
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="writer")
            ]
        },
        goto="supervisor",  # always return control to the supervisor after writing
    )
