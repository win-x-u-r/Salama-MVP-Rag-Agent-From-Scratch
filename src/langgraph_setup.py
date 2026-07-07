import logging
import time
from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from .backend_setup import llm
from .tools_setup import TOOLS
from .system_prompt_setup import build_system_prompt

logger = logging.getLogger("uvicorn.error")


def check_for_tools(state: MessagesState) -> Literal["tools", "responder"]:
    """Safely check if the agent successfully outputted a tool call."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Fallback safety: if the model somehow failed to output a tool call despite the constraint,
    # route straight to responder to prevent a hard graph crash.
    return "responder"


def retrieve_agent(state: MessagesState):
    """calls the retriever tool and returns the result to the responder agent"""
    try:
        t0 = time.perf_counter()
        system = SystemMessage(content=build_system_prompt())
        response = llm.bind_tools(TOOLS, tool_choice="any").invoke([system] + state["messages"])
        logger.info(f"[timing] retrieve agent (LLM): {time.perf_counter() - t0:.2f}s")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in retrieve agent: {e}")
        return {"messages": [AIMessage(content="An error occurred while processing your request, retreiver agent.")]}


def responder_agent(state:MessagesState):
    """writes the final answer to the user after the retriever agent has returned the results"""
    try:
        t0 = time.perf_counter()
        system = SystemMessage(content=build_system_prompt())
        payload = [system] + state["messages"]
        tokens = llm.get_num_tokens_from_messages(payload)
        response = llm.invoke(payload)
        logger.info(f"[timing] responder node: {time.perf_counter() - t0:.2f}s\n response: {response.content} \n tokens: {tokens}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in responder node: {e}")
        return {"messages": [AIMessage(content="An error occurred while compiling your answer, responder agent.")]}


builder = StateGraph(MessagesState)
builder.add_node("retrieve agent", retrieve_agent)
builder.add_node("responder agent", responder_agent)
builder.add_node("retriever", ToolNode(TOOLS))
builder.add_edge(START, "retrieve agent")
# builder.add_conditional_edges("agent", tools_condition)  # -> "tools" if a tool was called, else END
# builder.add_conditional_edges("retrieve agent", check_for_tools,{"tools": "tools", "responder": "responder agent"})  # -> "tools" if a tool was called, else END
builder.add_edge("retrieve agent", "retriever")
builder.add_edge("retriever", "responder agent")
builder.add_edge("responder agent", END)

checkpointer = InMemorySaver()  # short-term (conversation) memory, per thread
graph = builder.compile(checkpointer=checkpointer)

# Single local user
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
