"""
main.py — Entry point for the Content-Writing Multi-Agent System.

Run this file to start the workflow:

    python main.py

What happens end-to-end:
    1. ``validate_keys()``   → confirms GROQ_API_KEY and TAVILY_API_KEY are set.
    2. User prompt           → the human types a topic (e.g. "Quantum Computing").
    3. A default filename    → generated as  article_<random 4-digit int>.md
                               and embedded in the user request so the writer
                               knows what name to use.  NOTE: the tool name
                               'write_document' is intentionally NOT mentioned
                               in the user request — only the writer agent has
                               that tool, and mentioning it confuses the researcher
                               into trying to call it (causing a 400 error).
    4. ``build_workflow()``  → assembles the LangGraph StateGraph.
    5. ``app.stream()``      → kicks off the graph and streams step-by-step output.
    6. Supervisor decides    → routes to researcher first, then writer.
    7. Researcher searches   → calls Tavily, returns bullet-point findings.
    8. Writer drafts article → calls write_document, saves a .md file.
    9. Supervisor finishes   → routes to END; the article is in doc-result/.

Streaming output (formatted):
    - supervisor : prints the routing decision (researcher / writer / FINISH).
    - researcher : prints only the first 50 words so the terminal stays readable.
    - writer     : prints the full final message so you can confirm the save path.
    - other nodes: printed as-is.

Recursion limit:
    LangGraph's ``recursion_limit`` caps how many node executions are allowed
    in a single run.  This prevents infinite loops if the supervisor gets
    confused and keeps cycling between agents.
"""

import random                                                    # for generating unique filenames
from src.config.settings import validate_keys, RECURSION_LIMIT  # config & guard
from src.graph.workflow import build_workflow                    # graph factory


# ── Helper: pretty-print a single stream step ─────────────────────────────────
def _print_step(node: str, payload: dict) -> None:
    """Format and print one LangGraph stream step to the terminal.

    Different nodes get different treatment:
      - ``supervisor``  → print the routing decision (``next`` field only).
      - ``researcher``  → print only the first 50 words of the content so the
                          terminal is not flooded with the full research dump.
      - ``writer``      → print the full message so the user can see the article
                          summary and the file-save confirmation path.
      - other nodes     → print the raw payload dict.

    Args:
        node:    Name of the LangGraph node that just completed.
        payload: The state update dict returned by that node.
    """
    if node == "supervisor":
        # The supervisor payload contains a 'next' key with the routing decision
        next_step = payload.get("next", "?")
        print(f"  [supervisor] → routing to: {next_step}")

    elif node == "researcher":
        # Researcher output can be thousands of words; trim to first 50
        msgs = payload.get("messages", [])
        if msgs:
            # .content is the text string on a LangChain BaseMessage
            content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            words = content.split()
            # Join first 50 words; append "…" if the text was longer
            preview = " ".join(words[:50]) + (" …" if len(words) > 50 else "")
            print(f"  [researcher] → (preview) {preview}")
        else:
            print(f"  [researcher] → (no messages)")

    elif node == "writer":
        # Print the full writer response so the user can see the save path
        msgs = payload.get("messages", [])
        if msgs:
            content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            print(f"  [writer] →\n{content}")
        else:
            print(f"  [writer] → (no messages)")

    else:
        # Fallback for any future nodes added to the graph
        print(f"  [{node}] → {payload}")


# ── Main workflow function ─────────────────────────────────────────────────────
def run() -> None:
    """Validate configuration, collect user input, and execute the multi-agent workflow.

    Steps:
        1. Call ``validate_keys()`` to fail fast if API keys are absent.
        2. Prompt the user for an article topic.
        3. Generate a unique default filename  ``article_<NNNN>.md``  so the
           writer always has an explicit filename even if the LLM doesn't invent one.
        4. Build a structured user request string that embeds the filename.
        5. Compile the LangGraph workflow.
        6. Stream the graph execution, calling ``_print_step`` for each node.
        7. Print a completion banner when the workflow finishes.

    Returns:
        None.  The generated article is saved to the ``doc-result/`` folder
        as a side-effect of the writer agent's ``write_document`` tool call.

    Raises:
        EnvironmentError: Propagated from ``validate_keys()`` if any API key
                          is missing in the ``.env`` file.
    """
    # ── Step 1: Guard — ensure API keys are loaded ─────────────────────────────
    validate_keys()

    # ── Step 2: User Input ─────────────────────────────────────────────────────
    print("\n===== CONTENT WRITING MULTI-AGENT =====")
    topic = input("Enter the topic you want an article on: ").strip()

    # Exit gracefully if the user submits an empty string
    if not topic:
        print("No topic provided. Exiting.")
        return

    word_limit = input("Enter the desired word limit (e.g. 500): ").strip()
    if not word_limit:
        word_limit = "500"  # default to 500 if the user just presses Enter

    # ── Step 3: Generate a unique default filename ─────────────────────────────
    # random.randint(1000, 9999) gives a 4-digit suffix, e.g. "article_4273.md"
    # Embedding the filename in the user request means the writer agent receives
    # an explicit filename and won't invent its own (or forget to save at all).
    default_filename = f"article_{random.randint(1000, 9999)}.md"

    # ── Step 4: Build the user request message ─────────────────────────────────
    # This is the initial HumanMessage injected into the graph's message state.
    # IMPORTANT: Do NOT mention 'write_document' or any other tool name here.
    # The full user request is visible to ALL agents including the researcher.
    # If the researcher sees 'write_document' it will try to call it — but that
    # tool is only registered in the writer agent, causing a Groq 400 error.
    # The writer's own system prompt already instructs it to call write_document.
    user_request = (
        f"Research the topic '{topic}' thoroughly, then write a well-structured "
        f"Markdown article on it ensuring it is approximately {word_limit} words long, "
        f"and save it as '{default_filename}'."
    )

    # ── Step 5: Compile the LangGraph workflow ─────────────────────────────────
    app = build_workflow()

    # ── Step 6: Stream and print graph execution ───────────────────────────────
    print(f"\n----- WORKFLOW STARTED (output → doc-result/{default_filename}) -----\n")

    # app.stream() yields one dict per completed node execution.
    # Each dict has the shape: {node_name: state_update_payload}
    for step in app.stream(
        {"messages": [("user", user_request)]},  # seed state with user's request
        {"recursion_limit": RECURSION_LIMIT},     # safety cap (defined in settings.py)
    ):
        for node, payload in step.items():
            _print_step(node, payload)   # formatted output per node type
            print("---")

    # ── Step 7: Completion banner ──────────────────────────────────────────────
    print(f"\n===== WORKFLOW FINISHED =====")
    print(f"Article saved to: doc-result/{default_filename}")


# ── Script Guard ───────────────────────────────────────────────────────────────
# Ensures run() is only called when this file is executed directly,
# not when it is imported as a module.
if __name__ == "__main__":
    run()
