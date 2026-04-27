"""
researcher.py — Research Agent (part of the Research Agent Team).

Architecture (from the workflow diagram):
  User → Supervisor Agent → Research Agent Team → Supervisor Agent → …

The Research Agent Team consists of a single ReAct agent backed by the
Tavily web-search tool.  Its job is to:
  1. Receive a research topic from the supervisor via the shared message state.
  2. Autonomously call the Tavily search tool one or more times.
  3. Summarise the findings as concise bullet points with source URLs.
  4. Return control to the supervisor by calling ``researcher_node()``.

``create_react_agent`` (LangGraph prebuilt) builds a mini sub-graph that loops
between the LLM and the tool executor until the LLM stops emitting tool calls,
then returns the final response.
"""

from typing import Literal
from langchain_core.messages import AIMessage  # wraps agent output as an AI message
from langgraph.prebuilt import create_react_agent  # prebuilt ReAct agent sub-graph factory
from langgraph.types import Command               # tells the parent graph where to go next

from src.agent.llm import llm                    # shared Groq LLM instance
from src.tools.web_tools import tavily_tool      # Tavily web-search tool

# ── Researcher ReAct Agent ─────────────────────────────────────────────────────
# create_react_agent builds a LangGraph sub-graph that:
#   1. Calls the LLM with the current messages + a system prompt.
#   2. If the LLM emits a tool call → executes the tool and loops back to step 1.
#   3. If the LLM produces a final text answer → returns it.
#
# The system prompt is injected as a persistent instruction so the researcher
# always knows its role and output format regardless of conversation history.
researcher_agent = create_react_agent(
    llm,
    tools=[tavily_tool],  # the ONLY tool this agent is allowed to use
    prompt=(
        "You are a research assistant. Your ONLY job is to search the web and summarise findings. "
        "Use the Tavily search tool to gather accurate, up-to-date information on the user's topic. "
        "Summarize key findings as concise bullet points with source URLs. "
        "IMPORTANT CONSTRAINTS:\n"
        "  - Use ONLY the Tavily search tool. Do NOT call any other tool.\n"
        "  - Do NOT write, create, or save any files or documents.\n"
        "  - Do NOT ask follow-up questions. Return your research summary and stop."
    ),
)


# ── Researcher Node ────────────────────────────────────────────────────────────
def researcher_node(state) -> Command[Literal["supervisor"]]:
    """Execute the research agent and route the result back to the supervisor.

    This function is registered as the ``"researcher"`` node in the LangGraph
    ``StateGraph``.  It is called by LangGraph whenever the supervisor decides
    the next step is ``"researcher"``.

    Workflow:
        1. Pass the current shared ``state`` (all messages so far) to the
           researcher ReAct agent.
        2. Extract the agent's final text response (last message in the result).
        3. Wrap it in a ``HumanMessage`` tagged with ``name="researcher"`` so
           the supervisor can identify who produced this message.
        4. Return a ``Command`` that:
           - ``update``: appends the researcher's message to the shared state.
           - ``goto``:   always routes back to the supervisor for the next decision.

    Args:
        state: The current ``MessagesState`` dict shared across all graph nodes.
               Contains at minimum a ``"messages"`` key with the conversation so far.

    Returns:
        A ``Command`` object instructing LangGraph to update the state and
        move to the ``"supervisor"`` node next.

    Note:
        ``name="researcher"`` on the ``HumanMessage`` is important — the
        supervisor's prompt tells it to look at who sent each message when
        deciding the next step.
    """
    # Invoke the researcher ReAct sub-graph with the full shared state.
    # result["messages"] is the complete message history after the agent finishes.
    result = researcher_agent.invoke(state)

    return Command(
        update={
            # Append only the final assistant response; earlier intermediate
            # messages (tool calls, tool results) stay inside the sub-graph.
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",  # always hand control back to the supervisor
    )
