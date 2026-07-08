import logging
import time
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from .backend_setup import llm
from .tools_setup import TOOLS
from .system_prompt_setup import build_system_prompt

logger = logging.getLogger("uvicorn.error")

MAX_CLARIFY = 2  # loop guard: max requery/reprompt cycles per turn before forcing an answer


# ---- graph state: MessagesState + per-turn control fields ----
class ClarifierState(MessagesState):
    clarify_count: int   # how many clarify cycles have happened this turn
    mode: str            # "answer" (normal) | "ask" (responder should ask, not answer)


# ---- clarifier's structured decision (deterministic routing, not free text) ----
class ClarifierDecision(BaseModel):
    action: Literal["proceed", "requery", "reprompt"] = Field(
        description="proceed = question clear & passages adequate; "
        "requery = you can write a better search query yourself; "
        "reprompt = the user must supply missing details."
    )
    improved_query: str | None = Field(
        default=None,
        description="A better, more specific search query. Required when action='requery'.",
    )


CLARIFIER_PROMPT = (
    "You are the Clarifier in a RAG pipeline for GDRFA Dubai visa services.\n"
    "You are given the user's question, the search query the retriever used, and the retrieved "
    "passages. Choose EXACTLY ONE action:\n"
    "- 'proceed': the question is clear and the passages can answer it. This is the default — "
    "prefer it.\n"
    "- 'requery': the passages don't answer the question because the search query was too vague, "
    "too broad, or off-target, AND you can write a better, more specific query yourself. Put that "
    "query in 'improved_query'.\n"
    "- 'reprompt': the user's question is genuinely ambiguous or missing details that only the "
    "USER can provide (e.g. which visa type, their nationality / status / assets), so no query can "
    "answer it well.\n"
    "Only choose 'requery' or 'reprompt' when there is a real problem; otherwise 'proceed'."
)


def reset_turn(state: ClarifierState):
    """Runs once at the start of every turn so counters/flags never leak across turns."""
    return {"clarify_count": 0, "mode": "answer"}


def retrieve_agent(state: ClarifierState):
    """Forces a retrieval tool call so the pipeline always grounds its answer."""
    try:
        t0 = time.perf_counter()
        system = SystemMessage(content=build_system_prompt())
        response = llm.bind_tools(TOOLS, tool_choice="any").invoke([system] + state["messages"])
        logger.info(f"[timing] retrieve agent (LLM): {time.perf_counter() - t0:.2f}s")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in retrieve agent: {e}")
        return {"messages": [AIMessage(content="An error occurred while processing your request, retriever agent.")]}


def clarifier_agent(state: ClarifierState) -> Command[Literal["retrieve agent", "responder agent"]]:
    """Judges the query/retrieval and routes: proceed, requery, or reprompt the user."""
    count = state.get("clarify_count", 0)
    try:
        t0 = time.perf_counter()

        # Loop guard: after MAX_CLARIFY cycles, stop clarifying and just answer.
        if count >= MAX_CLARIFY:
            logger.info(f"[clarifier] cap reached ({count}) -> proceed")
            return Command(goto="responder agent")

        system = SystemMessage(content=CLARIFIER_PROMPT)
        decision = llm.with_structured_output(ClarifierDecision).invoke([system] + state["messages"])
        logger.info(
            f"[timing] clarifier node: {time.perf_counter() - t0:.2f}s "
            f"| action={decision.action} | count={count}"
        )

        if decision.action == "requery" and decision.improved_query:
            # Re-run retrieval with a better query (loop back), and count the cycle.
            logger.info(f"[clarifier] requery -> {decision.improved_query!r}")
            return Command(
                goto="retrieve agent",
                update={
                    "messages": [HumanMessage(content=f"(clarified query) {decision.improved_query}")],
                    "clarify_count": count + 1,
                },
            )

        if decision.action == "reprompt":
            # Ask the user for details. Flag the responder to ask (not answer); it streams as usual.
            logger.info("[clarifier] reprompt -> ask user for details")
            return Command(
                goto="responder agent",
                update={"mode": "ask", "clarify_count": count + 1},
            )

        # proceed
        return Command(goto="responder agent")
    except Exception as e:
        logger.error(f"Error in clarifier node: {e}")
        return Command(goto="responder agent")  # fail safe -> answer with what we have


def responder_agent(state: ClarifierState):
    """Streams the final answer, OR (mode='ask') a clarifying question back to the user."""
    try:
        t0 = time.perf_counter()
        mode = state.get("mode", "answer")
        prompt = build_system_prompt()
        if mode == "ask":
            prompt += (
                "\n\nIMPORTANT: The user's question is too vague or missing details to answer well. "
                "Do NOT attempt an answer. Instead, ask ONE short, specific clarifying question to "
                "get the missing detail (e.g. which visa type, their nationality, status, or assets)."
            )
        payload = [SystemMessage(content=prompt)] + state["messages"]
        tokens = llm.get_num_tokens_from_messages(payload)
        response = llm.invoke(payload)
        logger.info(
            f"[timing] responder node: {time.perf_counter() - t0:.2f}s | mode={mode} "
            f"| tokens: {tokens}\n response: {response.content}"
        )
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in responder node: {e}")
        return {"messages": [AIMessage(content="An error occurred while compiling your answer, responder agent.")]}


builder = StateGraph(ClarifierState)
builder.add_node("reset", reset_turn)
builder.add_node("retrieve agent", retrieve_agent)
builder.add_node("retriever", ToolNode(TOOLS))
builder.add_node("clarifier", clarifier_agent)
builder.add_node("responder agent", responder_agent)

builder.add_edge(START, "reset")
builder.add_edge("reset", "retrieve agent")
builder.add_edge("retrieve agent", "retriever")
builder.add_edge("retriever", "clarifier")
# "clarifier" routes itself via Command(goto=...) -> "retrieve agent" (requery) or "responder agent"
builder.add_edge("responder agent", END)

checkpointer = InMemorySaver()  # short-term (conversation) memory, per thread
graph = builder.compile(checkpointer=checkpointer)

# Single local user
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
