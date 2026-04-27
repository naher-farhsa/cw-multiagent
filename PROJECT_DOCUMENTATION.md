# Content-Writing Multi-Agent System: Architecture & Components

## Project Overview

The **Content-Writing Multi-Agent System** is a sophisticated LangGraph-based workflow that automates the end-to-end process of researching and writing articles. The system orchestrates multiple AI agents to collaborate on content creation: a Researcher gathers information from the web, and a Writer drafts, formats, and saves the final article. The entire workflow is managed by a central Supervisor that routes tasks between agents based on the conversation state.

### Key Objectives
- **Autonomous research** via web search (Tavily API)
- **Content generation** in Markdown format
- **Chart generation** for data visualization
- **Deterministic workflow** with LLM fallback routing
- **Safety guardrails** to prevent agent loops and hallucination

---

## Architecture Overview

```
START
  │
  ▼
Supervisor Agent ◄──────────────────────────────────┐
  │                                                  │
  ├─► Research Agent Team ──► (back to supervisor)  │
  │    └─ Tavily web-search tool                    │
  │                                                  │
  └─► Writing Agent Team ───► (back to supervisor) ─┘
       ├─ write_document
       ├─ read_document
       └─ generate_chart
        │
        ▼
      END
```

**Workflow Flow:**
1. User submits a topic and desired word count
2. Supervisor routes to **Researcher** → gathers findings via Tavily
3. Supervisor routes to **Writer** → drafts article, optionally creates charts, saves to disk
4. Supervisor routes to **END** → workflow completes

---

## Core LangGraph Components

### 1. **State** — Shared Data Container

**File:** `src/graph/supervisor.py`

```python
class State(MessagesState):
    """Extended graph state that tracks which node should execute next."""
    next: str  # populated by supervisor_node on every invocation
```

**Purpose:**
- Inherits from LangChain's `MessagesState`, which provides automatic message list management
- The `messages` field stores the conversation history (automatically reduced/appended)
- The `next` field tracks the supervisor's routing decision for observability

**Key Attributes:**
- `messages: list[BaseMessage]` — full conversation history with LLM-friendly reducer
- `next: str` — name of the next node to execute (e.g., "researcher", "writer", "FINISH")

**Why it matters:** All nodes share this single state dict, ensuring consistent information flow and preventing data silos.

---

### 2. **Nodes** — Executable Units of Work

Nodes are individual functions registered in the StateGraph that process the shared state.

#### **A. Supervisor Node**

**File:** `src/graph/supervisor.py` — `make_supervisor_node()` factory

**Responsibility:** Central orchestrator that decides which agent runs next.

**Routing Logic (Deterministic-First):**
```python
def supervisor_node(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    researcher_done = _agent_has_responded(state, "researcher")
    writer_done = _agent_has_responded(state, "writer")
    
    if not researcher_done:
        goto = "researcher"          # STEP 1: researcher hasn't run
    elif not writer_done:
        goto = "writer"              # STEP 2: writer hasn't run
    else:
        goto = END                   # STEP 3: both done → FINISH
    
    return Command(goto=goto, update={"next": goto})
```

**Key Features:**
- **Deterministic guardrails** prevent LLM hallucination by programmatically checking who has already responded
- **LLM fallback** (with structured output schema) for future extensibility if more agents are added
- **System prompt** enforces the exact three-step sequence

**Output Format:**
- Returns a `Command` object with:
  - `goto`: next node name or END
  - `update`: state dict containing the routing decision

---

#### **B. Researcher Node**

**File:** `src/agent/researcher.py` — `researcher_node()`

**Responsibility:** Autonomously search the web for topic information.

**Architecture:**
- **Sub-graph:** Uses LangGraph's `create_react_agent()` (ReAct pattern)
- **Tools:** Only has access to `tavily_tool` (web search)
- **System Prompt:** Instructs the agent to summarize findings as bullet points with URLs

**Execution Flow:**
1. Receives the user's topic from shared state
2. Calls Tavily search tool (can loop multiple times if needed)
3. Summarizes findings
4. Returns a `Command(goto="supervisor")` with the research summary wrapped in a `HumanMessage(name="researcher")`

**Code Example:**
```python
def researcher_node(state) -> Command[Literal["supervisor"]]:
    result = researcher_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )
```

---

#### **C. Writer Node**

**File:** `src/agent/writer.py` — `writer_node()`

**Responsibility:** Drafts a Markdown article and saves it to disk.

**Architecture:**
- **Sub-graph:** ReAct agent with three tools
- **Tools:**
  - `write_document` — saves Markdown files
  - `read_document` — reads saved drafts for review
  - `generate_chart` — creates matplotlib visualizations
- **System Prompt:** Instructs the agent on the three roles (writer, note-taker, chart-generator)

**Typical Execution:**
1. Receives researcher findings in the message state
2. Optionally generates a chart if data warrants visualization
3. Drafts a well-structured Markdown article
4. Calls `write_document()` to persist to `doc-result/<filename>.md`
5. Returns confirmation message and routes back to supervisor

---

### 3. **Edges** — Routes Between Nodes

**File:** `src/graph/workflow.py`

```python
builder = StateGraph(State)

# Register nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

# Explicit entry point
builder.add_edge(START, "supervisor")

# Implicit routing via Command objects (researcher → supervisor, writer → supervisor)
# No explicit edges needed; Command(goto=...) carries the routing instruction

app = builder.compile()
```

**Types of Edges:**
1. **Unconditional (`add_edge`):** `START → supervisor` — always execute supervisor first
2. **Implicit (via `Command`):** Nodes return routing instructions embedded in their return values
3. **Conditional (not used here):** Could use `add_conditional_edges()` for complex branching

**Why Command over add_edge:**
- Allows nodes to make dynamic routing decisions based on state
- Keeps routing logic inside node functions (cleaner separation of concerns)
- Enables LangGraph to handle tool calls within sub-graphs transparently

---

### 4. **START & END** — Graph Boundaries

**File:** `src/graph/workflow.py`

```python
from langgraph.graph import StateGraph, START

builder.add_edge(START, "supervisor")  # entry point
# When supervisor returns Command(goto=END), the graph terminates
```

**Semantics:**
- **START:** Sentinel value representing the graph's entry point (always `supervisor`)
- **END:** Sentinel value that terminates the workflow
- When any node returns `Command(goto=END)` or `goto="__end__"`, the graph stops executing and returns the final state

---

### 5. **Orchestrator / Router** — The Supervisor

The supervisor is the central orchestrator with a **two-tier routing strategy**:

#### **Tier 1: Deterministic Guardrails**

```python
def _agent_has_responded(state: State, agent_name: str) -> bool:
    """Check if a message with name == agent_name exists in the state."""
    for msg in state.get("messages", []):
        if getattr(msg, "name", None) == agent_name:
            return True
    return False
```

**Why deterministic first:**
- Groq's Llama-3.3 frequently ignored prompt instructions about sequencing
- Would loop indefinitely on "researcher" without calling "writer"
- Programmatic checks eliminate LLM hallucination risk

#### **Tier 2: LLM Fallback (for extensibility)**

If more agents are added in the future, the supervisor falls back to an LLM-based decision with structured output:

```python
class Router(TypedDict):
    next: Literal["FINISH", "researcher", "writer"]

llm.with_structured_output(Router).invoke(messages)
```

---

### 6. **Loop** — ReAct Sub-graphs

Within each worker node (researcher, writer), there is an internal **ReAct loop**:

**ReAct Pattern (inside researcher_agent / writer_agent):**
```
LLM thinks
   ↓
LLM decides: call tool? or return final answer?
   ↓
If tool call → execute tool
   ↓
Append tool result to messages
   ↓
Loop back to LLM
   ↓
If final answer → return and exit loop
```

**Implementation:**
```python
researcher_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt="You are a research assistant..."
)
```

LangGraph's `create_react_agent()` handles the loop internally — no explicit while loop needed.

---

## Additional Elements & Tools

### 7. **Tool Functions** — Agent Capabilities

**File:** `src/tools/file_tools.py`, `src/tools/web_tools.py`

#### **A. Web Tools**
```python
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(max_results=3)
```
- **Purpose:** Enables researcher to search the web
- **Max results:** Kept small (3) to balance relevance with token usage
- **Returns:** Structured search results with URLs and summaries

#### **B. File Tools**

**`write_document(content: str, file_name: str) → str`**
- Saves Markdown content to `doc-result/<file_name>`
- UTF-8 encoding for international characters
- Returns confirmation path

**`read_document(file_name: str, start: Optional[int], end: Optional[int]) → str`**
- Reads a previously saved document
- Supports line-range slicing
- Allows writer to review/extend drafts

**`generate_chart(title: str, labels: List[str], values: List[float], chart_type: str) → str`**
- Creates matplotlib bar/line/pie charts
- Saves as `.png` to `doc-result/`
- Writer can embed via Markdown: `![title](doc-result/chart.png)`

---

### 8. **LLM Instance** — Shared Model

**File:** `src/agent/llm.py`

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="mixtral-8x7b-32768")  # or gpt-4o via OpenAI
```

**Why shared:**
- Single model instance used by supervisor, researcher, and writer
- Reduces latency (connection pooling)
- Consistent behavior across all agents

---

### 9. **Configuration & Settings**

**File:** `src/config/settings.py`

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "doc-result"
OUTPUT_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
RECURSION_LIMIT = 30

def validate_keys() -> None:
    """Fail fast if required API keys are missing."""
    required = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise EnvironmentError(f"Missing: {', '.join(missing)}")
```

**Key Settings:**
- **OUTPUT_DIR:** Where articles are saved
- **RECURSION_LIMIT:** Safety cap to prevent infinite loops (LangGraph's built-in guard)
- **API Key validation:** Called at startup to fail fast if credentials are missing

---

### 10. **Command Objects** — Routing Instructions

**Concept from LangGraph's `langgraph.types`:**

```python
from langgraph.types import Command

# Inside a node:
return Command(
    goto="next_node_name",              # where to go next
    update={"messages": [...]}          # state updates to apply
)
```

**Advantages:**
- Encapsulates routing logic inside the node
- Allows dynamic decisions based on state
- Supports both unconditional and conditional routing

---

### 11. **Message Types** — LangChain Integration

**Files:** Imported from `langchain_core.messages`

```python
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,     # user input
    AIMessage,        # LLM output
    ToolMessage,      # tool result
    SystemMessage     # system prompt
)
```

**Usage in this project:**
- **HumanMessage(name="researcher"):** Wraps researcher output so supervisor recognizes its origin
- **AIMessage(name="writer"):** Wraps writer output with the agent's name
- **ToolMessage:** Internal (handled by ReAct agent); contains tool result content
- **SystemMessage:** Contains agent role descriptions and instructions

---

### 12. **Structured Output** — Type-Safe LLM Responses

**Pattern used in supervisor fallback:**

```python
from typing_extensions import TypedDict

class Router(TypedDict):
    next: Literal["FINISH", "researcher", "writer"]

response = llm.with_structured_output(Router).invoke(messages)
# LLM is forced to return a dict matching the Router schema
```

**Benefit:** Eliminates parsing errors and ensures the LLM's response conforms to a known shape.

---

### 13. **State Reducers** — Message Management

**From LangChain's `MessagesState`:**

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    """Inherits automatic message reduction."""
    next: str
```

**The reducer:**
- Automatically appends new messages instead of overwriting
- Preserves the full conversation history
- Prevents accidental loss of context

---

### 14. **Compiled Graph** — Executable Artifact

**Created by `build_workflow()` in `src/graph/workflow.py`:**

```python
app = builder.compile()
# app is now a CompiledGraph object, ready to stream or invoke

for step in app.stream(
    {"messages": [("user", "Write an article about AI")]},
    {"recursion_limit": 30}
):
    print(step)  # each dict represents one node's output
```

**Properties:**
- Validates the graph structure at compile time
- Exposes `.stream()` and `.invoke()` methods for execution
- Supports configuration like `recursion_limit` at runtime

---

### 15. **Streaming & Step Formatting** — User-Facing Output

**File:** `main.py`

```python
def _print_step(node: str, payload: dict) -> None:
    """Format node output for terminal display."""
    if node == "supervisor":
        next_step = payload.get("next", "?")
        print(f"  [supervisor] → routing to: {next_step}")
    elif node == "researcher":
        # Print only first 50 words (research can be verbose)
        content = payload["messages"][-1].content
        preview = " ".join(content.split()[:50]) + " …"
        print(f"  [researcher] → {preview}")
    elif node == "writer":
        # Print full output including save path
        print(f"  [writer] →\n{payload['messages'][-1].content}")

for step in app.stream(initial_state, config):
    for node, payload in step.items():
        _print_step(node, payload)
        print("---")
```

**Purpose:** Makes the graph execution visible and debuggable.

---

## Data Flow & Message Passing

### **Example Execution Trace**

```
1. START
   ↓
2. supervisor_node(state)
   state.messages = [("user", "Research AI trends, write article")]
   → Checks: researcher_done? No
   → goto = "researcher"
   → Command(goto="researcher", update={"next": "researcher"})
   ↓
3. researcher_node(state)
   → researcher_agent.invoke(state)
   → Calls tavily_tool multiple times (internal loop)
   → Returns AI research summary
   → Wraps in HumanMessage(name="researcher", content=summary)
   → state.messages.append(researcher_message)
   → Command(goto="supervisor", update={"messages": [researcher_message]})
   ↓
4. supervisor_node(state)
   state.messages = [..., HumanMessage(name="researcher", ...)]
   → Checks: researcher_done? Yes (found message with name="researcher")
   → Checks: writer_done? No
   → goto = "writer"
   → Command(goto="writer", update={"next": "writer"})
   ↓
5. writer_node(state)
   → writer_agent.invoke(state)  [sub-graph with ReAct loop]
   → Optionally calls generate_chart()
   → Calls write_document() → saves article to doc-result/article_1234.md
   → Returns AIMessage(name="writer", content="Saved to doc-result/article_1234.md")
   → state.messages.append(writer_message)
   → Command(goto="supervisor", update={"messages": [writer_message]})
   ↓
6. supervisor_node(state)
   → Checks: researcher_done? Yes
   → Checks: writer_done? Yes (found message with name="writer")
   → goto = END
   → Command(goto=END, update={"next": END})
   ↓
7. END
   Graph terminates; final state is returned to main.py
   Article is now on disk in doc-result/
```

---

## Safety & Guardrails

### **1. Recursion Limit**
```python
for step in app.stream(state, {"recursion_limit": 30}):
    ...
```
- Prevents infinite loops if supervisor gets stuck
- Default cap: 30 node executions

### **2. Deterministic Routing**
- Programmatic checks prevent LLM hallucination
- If supervisor somehow returns an invalid node name, Command validation fails at compile time

### **3. Tool Restrictions**
- **Researcher:** Only has access to `tavily_tool` (can't call write_document)
- **Writer:** Has file tools and chart generation (can't call web search)
- Prevents accidental misuse or confusion

### **4. System Prompts**
- Each agent receives explicit role descriptions
- "Do NOT ask follow-up questions" → forces autonomous action
- "IMPORTANT CONSTRAINTS" section lists forbidden actions

---

## Project Structure Summary

```
cw-multiagent/
├── main.py                 # Entry point; orchestrates the workflow execution
├── .env                    # API keys (GROQ_API_KEY, TAVILY_API_KEY)
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py     # API keys, OUTPUT_DIR, validation
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── llm.py          # Shared ChatGroq instance
│   │   ├── researcher.py   # researcher_node + ReAct sub-graph
│   │   └── writer.py       # writer_node + ReAct sub-graph
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── supervisor.py   # State schema + make_supervisor_node() factory
│   │   └── workflow.py     # build_workflow() assembles the full StateGraph
│   │
│   └── tools/
│       ├── __init__.py
│       ├── web_tools.py    # tavily_tool
│       └── file_tools.py   # write_document, read_document, generate_chart
│
└── doc-result/             # Output directory (created at runtime)
    └── article_*.md        # Generated articles
```

---

## How to Run

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY=gsk_...
#   TAVILY_API_KEY=tvly-...

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the workflow
python main.py

# You'll be prompted:
#   Enter the topic you want an article on: Quantum Computing
#   Enter the desired word limit (e.g. 500): 1000

# 4. Watch the graph execute and view the output
# Article saved to: doc-result/article_1234.md
```

---

## Key Takeaways

| Component | Purpose | Key File(s) |
|-----------|---------|-----------|
| **State** | Shared data container with message history | `src/graph/supervisor.py` |
| **Nodes** | Executable units (supervisor, researcher, writer) | `src/agent/researcher.py`, `src/agent/writer.py`, `src/graph/supervisor.py` |
| **Edges** | Routes between nodes; mostly implicit via Command | `src/graph/workflow.py` |
| **START / END** | Graph entry and exit points | `src/graph/workflow.py` |
| **Orchestrator** | Central router with deterministic guardrails | `src/graph/supervisor.py` |
| **ReAct Loops** | Sub-graph tool-calling loops inside workers | `src/agent/researcher.py`, `src/agent/writer.py` |
| **Tools** | LLM-callable functions (web search, file I/O, charts) | `src/tools/web_tools.py`, `src/tools/file_tools.py` |
| **LLM** | Shared model instance (Groq or OpenAI) | `src/agent/llm.py` |
| **Config** | Settings, API keys, output directory | `src/config/settings.py` |
| **Command** | Dynamic routing instructions from nodes | Imported from `langgraph.types` |
| **Message Types** | LangChain message wrappers for context | Imported from `langchain_core.messages` |
| **Streaming** | Real-time execution with formatted output | `main.py` |

---

## Conclusion

The Content-Writing Multi-Agent System demonstrates a production-ready implementation of multi-agent orchestration using LangGraph. Its **deterministic-first routing**, **tool-based architecture**, and **clear separation of concerns** make it both robust and extensible. The system showcases best practices in prompt engineering, state management, and error handling for autonomous AI workflows.
