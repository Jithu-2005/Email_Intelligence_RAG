from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config.config import llm
from business_logic.email_logic import SYSTEM_PROMPT
from tools.search_tool import search_tool
from tools.summarize_tool import summarize_tool
from tools.extract_tool import extract_tool
from tools.draft_tool import draft_tool

tools = [search_tool, summarize_tool, extract_tool, draft_tool]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def call_model(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app_graph = graph.compile()
