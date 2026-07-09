import logging
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from .backend_setup import llm
from .tools_setup import TOOLS, retrieve
from .system_prompt_setup import build_system_prompt, NO_THINK

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


def first_retrieve(state: ClarifierState):
    """First-pass retrieval without an LLM: search with the user's own words.

    Saves the 1.5-4s query-writing LLM call on every turn; when the naive
    query retrieves poorly, the clarifier's 'requery' loop brings the LLM
    back in via `retrieve agent`.
    """
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = str(msg.content)
            break
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "retrieve",
                    "args": {"query": question},
                    "id": f"first-retrieve-{uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )
        ]
    }


def retrieve_agent(state: ClarifierState):
    """Forces a retrieval tool call so the pipeline always grounds its answer."""
    try:
        t0 = time.perf_counter()
        system = SystemMessage(content=build_system_prompt())
        response = llm.bind_tools([retrieve], tool_choice="retrieve").invoke([system] + state["messages"])
        logger.info(f"[timing] retrieve agent (LLM): {time.perf_counter() - t0:.2f}s")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in retrieve agent: {e}")
        return {"messages": [AIMessage(content="An error occurred while processing your request, retriever agent.")]}


def _clarifier_view(state: ClarifierState) -> list:
    """Compact clarifier input: question + search query + truncated passages.

    Passing full history (with every past retrieval result) made the
    clarifier's prompt grow each turn; it only needs this turn's evidence.
    """
    question, search_query, passages = "", "", ""
    for msg in reversed(state["messages"]):
        if not passages and isinstance(msg, ToolMessage):
            passages = str(msg.content)[:2000]
        elif not search_query and isinstance(msg, AIMessage) and msg.tool_calls:
            search_query = msg.tool_calls[0]["args"].get("query", "")
        elif isinstance(msg, HumanMessage):
            question = str(msg.content)
            break
    return [
        HumanMessage(
            content=(
                f"USER QUESTION: {question}\n\n"
                f"SEARCH QUERY USED: {search_query}\n\n"
                f"RETRIEVED PASSAGES (truncated):\n{passages}"
            )
        )
    ]


def clarifier_agent(state: ClarifierState) -> Command[Literal["retrieve agent", "responder agent"]]:
    """Judges the query/retrieval and routes: proceed, requery, or reprompt the user."""
    count = state.get("clarify_count", 0)
    try:
        t0 = time.perf_counter()

        # Loop guard: after MAX_CLARIFY cycles, stop clarifying and just answer.
        if count >= MAX_CLARIFY:
            logger.info(f"[clarifier] cap reached ({count}) -> proceed")
            return Command(goto="responder agent")

        system = SystemMessage(content=CLARIFIER_PROMPT + NO_THINK)
        decision = llm.with_structured_output(ClarifierDecision).invoke([system] + _clarifier_view(state))
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
        response = llm.invoke(payload)
        # token counts come free & exact from the provider's response;
        # get_num_tokens_from_messages cost ~1.7s of GPT-2 tokenizing per turn
        usage = getattr(response, "usage_metadata", None) or {}
        logger.info(
            f"[timing] responder node: LLM={time.perf_counter() - t0:.2f}s | mode={mode} "
            f"| tokens: in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}\n"
        )
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in responder node: {e}")
        return {"messages": [AIMessage(content="An error occurred while compiling your answer, responder agent.")]}


builder = StateGraph(ClarifierState)
builder.add_node("reset", reset_turn)
builder.add_node("first retrieve", first_retrieve)
builder.add_node("retrieve agent", retrieve_agent)
builder.add_node("retriever", ToolNode(TOOLS))
builder.add_node("clarifier", clarifier_agent)
builder.add_node("responder agent", responder_agent)

builder.add_edge(START, "reset")
builder.add_edge("reset", "first retrieve")       # first pass: no LLM
builder.add_edge("first retrieve", "retriever")
builder.add_edge("retrieve agent", "retriever")   # requery path keeps the LLM
builder.add_edge("retriever", "clarifier")
# "clarifier" routes itself via Command(goto=...) -> "retrieve agent" (requery) or "responder agent"
builder.add_edge("responder agent", END)

checkpointer = InMemorySaver()  # short-term (conversation) memory, per thread
graph = builder.compile(checkpointer=checkpointer)

# Single local user
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
