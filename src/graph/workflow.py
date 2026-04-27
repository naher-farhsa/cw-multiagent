from langgraph.graph import StateGraph, START

from src.agent.llm import llm
from src.agent.researcher import researcher_node
from src.agent.writer import writer_node
from src.graph.supervisor import State, make_supervisor_node


def build_workflow():
    supervisor_node = make_supervisor_node(llm, ["researcher", "writer"])

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "supervisor")

    return builder.compile()
