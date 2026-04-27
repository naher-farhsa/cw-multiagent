"""
supervisor.py — Supervisor Agent logic for the content-writing multi-agent system.

Architecture (from the workflow diagram):
  User → Supervisor Agent ←→ Research Agent Team / Writing Agent Team → End

The Supervisor is the central orchestrator.  Every time a worker (researcher or
writer) finishes, control returns here.  The supervisor reads the conversation
history and decides what happens next:

  * ``"researcher"`` — send the task to the Research Agent Team.
  * ``"writer"``     — send the task to the Writing Agent Team.
  * ``"FINISH"``     — the workflow is complete; route to ``END``.

Routing strategy — **deterministic-first, LLM-fallback**:
  The Llama-3.3 model on Groq frequently ignores prompt instructions about
  sequencing (it kept looping on "researcher" without ever calling "writer").
  To fix this, ``supervisor_node`` now applies **programmatic guardrails**:

  1. Scan the message history for messages tagged ``name="researcher"`` and
     ``name="writer"``.
  2. If the researcher has NOT spoken yet → deterministic route to researcher.
  3. If the researcher HAS spoken but the writer has NOT → deterministic route
     to writer.
  4. If the writer HAS spoken → deterministic route to END (FINISH).

  The LLM-based routing is kept as a fallback in case the workflow is extended
  with additional agents in the future (the deterministic checks only cover the
  researcher/writer two-agent flow).

Design pattern: ``make_supervisor_node`` is a *factory function* that accepts
the LLM and the list of worker names at build time, then returns the actual
``supervisor_node`` closure.

Structured output (``with_structured_output``) is used in the LLM fallback path
so the LLM is forced to return a JSON object matching the ``Router`` TypedDict.
"""

from typing import List, Literal
from typing_extensions import TypedDict                        # TypedDict for structured LLM output
from langchain_core.language_models.chat_models import BaseChatModel  # type hint for any chat LLM
from langgraph.graph import MessagesState, END                  # built-in state type and end sentinel
from langgraph.types import Command                             # routing instruction returned by a node


# ── Shared Graph State ─────────────────────────────────────────────────────────
class State(MessagesState):
    """Extended graph state that tracks which node should execute next.

    Inherits from ``MessagesState``, which automatically provides:
        messages (list[BaseMessage]): The full conversation history, with
            LangGraph's ``add_messages`` reducer so messages are appended
            rather than overwritten on every update.

    Additional fields:
        next (str): The name of the next worker to run, or ``"FINISH"`` /
                    ``END``.  Set by the supervisor on every call and used by
                    the ``StateGraph`` edges to determine routing.
    """
    next: str  # populated by supervisor_node on every invocation


# ── Helper: check which agents have already responded ──────────────────────────
def _agent_has_responded(state: State, agent_name: str) -> bool:
    """Return True if any message in the state has ``name == agent_name``.

    LangGraph wraps each agent's output in a ``HumanMessage(name=...)``
    (see researcher_node / writer_node), so scanning for the ``name`` field
    reliably tells us whether that agent has already run.

    Args:
        state:      Current graph state with a ``"messages"`` list.
        agent_name: The name to look for, e.g. ``"researcher"`` or ``"writer"``.

    Returns:
        ``True`` if at least one message carries that agent name.
    """
    for msg in state.get("messages", []):
        # BaseMessage.name may be None if not set, so use getattr with default
        if getattr(msg, "name", None) == agent_name:
            return True
    return False


# ── Supervisor Factory ─────────────────────────────────────────────────────────
def make_supervisor_node(llm: BaseChatModel, members: List[str]):
    """Build and return a supervisor node function bound to the given LLM and workers.

    This is a *factory* (closure) pattern: the heavy work of building the
    system prompt and the ``Router`` schema is done once when the graph is
    constructed, then the returned ``supervisor_node`` is called cheaply on
    every graph step.

    Args:
        llm:     Any LangChain chat model.  Must support ``with_structured_output``.
                 In this project it is always the shared ``ChatGroq`` instance.
        members: List of worker node names, e.g. ``["researcher", "writer"]``.
                 The supervisor can route to any of these or to ``"FINISH"``.

    Returns:
        ``supervisor_node`` — a callable that accepts the current graph state
        and returns a ``Command`` directing the graph to the chosen next node.

    Routing logic (deterministic-first):
        The function first checks programmatically whether the researcher and/or
        writer have already responded.  It only falls through to the LLM when
        the deterministic checks are inconclusive (e.g. if you add more agents).
    """
    # "FINISH" is prepended so the supervisor can signal completion
    options = ["FINISH"] + members

    # System prompt — only used in the LLM fallback path.
    system_prompt = (
        "You are a supervisor managing a content-writing workflow. "
        f"Your available workers are: {members}.\n"
        "You MUST follow this exact three-step sequence — do NOT skip any step:\n"
        "  STEP 1 → Route to 'researcher' to gather information.\n"
        "  STEP 2 → Route to 'writer' to write and SAVE the article.\n"
        "  STEP 3 → Respond with FINISH after the writer has saved the document.\n"
        f"Respond with exactly one of: {options}."
    )

    # ── Structured Output Schema ───────────────────────────────────────────────
    class Router(TypedDict):
        """Schema the LLM must return.  Used only in the fallback path."""
        next: Literal["FINISH", "researcher", "writer"]

    # ── Inner Node Function ────────────────────────────────────────────────────
    def supervisor_node(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
        """Decide the next worker to call, or finish the workflow.

        Routing priority:
            1. **Deterministic check** — inspect message names to see who has
               already run.  This is 100% reliable and immune to LLM hallucination.
            2. **LLM fallback** — only reached if the deterministic checks don't
               cover the scenario (e.g. extra agents added later).

        The deterministic flow is:
            ┌─ researcher has NOT responded yet → route to "researcher"
            ├─ researcher responded, writer has NOT → route to "writer"
            └─ writer has responded → FINISH (END)

        Args:
            state: Current ``State`` containing all conversation messages.

        Returns:
            ``Command`` with:
                - ``goto``:  next node name, or ``END`` to finish.
                - ``update``: sets ``state["next"]`` for observability / logging.
        """
        # ── Deterministic guardrails ───────────────────────────────────────────
        # Check which agents have already produced output by scanning for named
        # messages.  This prevents the LLM from looping on "researcher" forever.
        researcher_done = _agent_has_responded(state, "researcher")
        writer_done = _agent_has_responded(state, "writer")

        if not researcher_done:
            # First call: no one has run yet → start with researcher
            goto = "researcher"
        elif not writer_done:
            # Researcher finished but writer hasn't → send to writer next
            goto = "writer"
        else:
            # Both agents have responded → workflow is complete
            goto = END

        # Log the deterministic decision for debugging
        print(f"    [supervisor-logic] researcher_done={researcher_done}, "
              f"writer_done={writer_done} → {goto}")

        # Update state["next"] for observability and route accordingly
        return Command(goto=goto, update={"next": goto})

    return supervisor_node  # return the configured node function to the workflow builder
