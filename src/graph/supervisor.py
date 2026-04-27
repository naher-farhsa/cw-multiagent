from typing import List, Literal
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState, END
from langgraph.types import Command


class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: List[str]):
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor managing a content-writing workflow with the following "
        f"workers: {members}. Given the user's request, decide which worker should act next. "
        "Typical flow: 'researcher' first to gather facts, then 'writer' to produce and save "
        "the final article. Once the writer has saved the document, respond with FINISH. "
        f"Respond with exactly one of: {options}."
    )

    class Router(TypedDict):
        next: Literal["FINISH", "researcher", "writer"]

    def supervisor_node(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    return supervisor_node
