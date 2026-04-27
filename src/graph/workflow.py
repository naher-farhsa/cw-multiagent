"""
workflow.py — Assembles the full LangGraph StateGraph for the content-writing system.

This is the only file that knows about all three nodes and wires them together.
Every other module is concerned with a single responsibility; this module
combines them into the executable graph.

Graph topology (mirrors the workflow diagram):

    START
      │
      ▼
  supervisor  ──────────────────────────────────────┐
      │                                             │
      ├──► researcher ──► (back to supervisor)      │  FINISH → END
      │                                             │
      └──► writer     ──► (back to supervisor) ─────┘

Key LangGraph concepts used here:
  * ``StateGraph(State)``  — creates a graph whose nodes share the ``State``
                             schema (messages + next field).
  * ``add_node``           — registers a callable (node function) under a name.
  * ``add_edge(START, …)`` — sets the unconditional entry point of the graph.
  * ``builder.compile()``  — validates the graph structure and returns a
                             runnable ``CompiledGraph`` (called ``app``).

The ``researcher_node`` and ``writer_node`` both return ``Command(goto="supervisor")``,
so no explicit edges from them to the supervisor are needed — the ``Command``
object carries the routing instruction inside the return value.
"""

from langgraph.graph import StateGraph, START  # graph builder and entry-point sentinel

from src.agent.llm import llm                        # shared Groq LLM (passed to supervisor)
from src.agent.researcher import researcher_node     # Research Agent Team node
from src.agent.writer import writer_node             # Writing Agent Team node
from src.graph.supervisor import State, make_supervisor_node  # state schema + supervisor factory


def build_workflow():
    """Construct and compile the multi-agent LangGraph workflow.

    This function:
        1. Creates the supervisor node via the factory (binds LLM + worker list).
        2. Instantiates a ``StateGraph`` using the shared ``State`` schema.
        3. Registers all three nodes: supervisor, researcher, writer.
        4. Adds the single unconditional entry edge: START → supervisor.
        5. Compiles and returns the executable graph.

    Why is routing not explicitly defined for researcher/writer?
        Both worker nodes return ``Command(goto="supervisor")``, which is
        LangGraph's way of embedding routing inside the node's return value
        rather than declaring static edges.  Only the START → supervisor
        edge needs to be explicit.

    Returns:
        ``CompiledGraph``: A runnable LangGraph object that accepts an initial
        state dict and can be iterated via ``.stream()`` or ``.invoke()``.

    Usage::

        app = build_workflow()
        for step in app.stream(
            {"messages": [("user", "Write an article about AI.")]},
            {"recursion_limit": 30},
        ):
            print(step)
    """
    # ── Build the supervisor node ──────────────────────────────────────────────
    # The factory returns a closure already configured with the LLM and the
    # list of valid workers.  This keeps supervisor.py generic.
    supervisor_node = make_supervisor_node(llm, ["researcher", "writer"])

    # ── Assemble the StateGraph ────────────────────────────────────────────────
    # StateGraph(State) creates a graph where every node shares the same
    # State dict (messages list + next field).
    builder = StateGraph(State)

    # Register each node under a string key.  The key is what Command(goto=...)
    # and add_edge() reference when routing.
    builder.add_node("supervisor", supervisor_node)   # central orchestrator
    builder.add_node("researcher", researcher_node)  # Research Agent Team
    builder.add_node("writer", writer_node)          # Writing Agent Team

    # Set the graph entry point: the very first node called is always "supervisor"
    builder.add_edge(START, "supervisor")

    # ── Compile ───────────────────────────────────────────────────────────────
    # compile() validates the graph (checks for unreachable nodes, missing edges)
    # and returns a CompiledGraph ready to run.
    return builder.compile()
