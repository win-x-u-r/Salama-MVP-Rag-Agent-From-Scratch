import logging
import re
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from .backend_setup import llm, clarifier_llm, store as memory_store
from .tools_setup import TOOLS
from .system_prompt_setup import build_system_prompt, NO_THINK

logger = logging.getLogger("uvicorn.error")

MAX_CLARIFY = 2  # loop guard: max requery/reprompt cycles per turn before forcing an answer

# Single local user for now; swap for an id from AG-UI forwardedProps for multi-user.
MEMORY_USER_ID = "local-user"
FACTS_NAMESPACE = ("users", MEMORY_USER_ID, "facts")
THREADS_NAMESPACE = ("users", MEMORY_USER_ID, "threads")  # conversation titles/recency


# ---- graph state: MessagesState + per-turn control fields ----
class ClarifierState(MessagesState):
    clarify_count: int   # how many clarify cycles have happened this turn
    mode: str            # "answer" (normal) | "ask" (responder should ask, not answer)
    improved_query: str  # set by the clarifier on requery; consumed by issue_retrieve
    memory_erased: bool  # facts were wiped this turn -> responder confirms it
    recalled_facts: str  # filled by the parallel memory-recall branch


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
    "- 'proceed': the passages contain ANY information that directly answers the question. This "
    "is the default — prefer it. Do NOT requery to narrow down or refine a question the passages "
    "already answer.\n"
    "- 'requery': the passages are clearly off-topic or empty for this question, AND you can "
    "write a better, more specific query yourself. Put that query in 'improved_query'.\n"
    "- 'reprompt': the user's question is genuinely ambiguous or missing details that only the "
    "USER can provide (e.g. which visa type, their nationality / status / assets), so no query can "
    "answer it well.\n"
    "Only choose 'requery' or 'reprompt' when there is a real problem; otherwise 'proceed'."
)


def reset_turn(state: ClarifierState):
    """Runs once at the start of every turn so counters/flags never leak across turns."""
    return {
        "clarify_count": 0,
        "mode": "answer",
        "improved_query": "",
        "memory_erased": False,
        "recalled_facts": "",
    }



_PUNCT = r"[\s!.,؟?،]*"
_SKIP_RETRIEVAL_PATTERNS = [
    # -- small talk / acknowledgements --
    (re.compile(
        r"^(hi+|hello|hey+|yo|good (morning|afternoon|evening)|how are you\??|"
        r"how's it going\??|thanks?( you)?( so much)?|thank you|ok(ay)?( thanks?)?|"
        r"got it|cool|nice|great|perfect|awesome|bye|goodbye|see you|"
        r"salam|hala+|marhaba|"
        r"مرحبا+|اهلا+|أهلا+|هلا+|السلام عليكم( ورحمة الله( وبركاته)?)?|"
        r"صباح الخير|مساء الخير|كيف حالك؟?|شكرا+( جزيلا)?( لك)?|تمام( شكرا)?|"
        r"ممتاز|رائع|حلو|مع السلامة|وداعا)"
        + _PUNCT + r"$",
        re.IGNORECASE,
    ), 40),
    # -- about the assistant (capabilities / identity / help) --
    (re.compile(
        r"^((what|who) are you\??|who r u|what can you (do|help( me)? with)( for me)?\??|"
        r"what do you do\??|how can you help( me)?\??|"
        r"what (kind of )?(things|questions|stuff) can (you|u) (do|answer|help with)\??|"
        r"help( me)?|tell me (about yourself|what you can do)|"
        r"what is this( chatbot| assistant| app)?\??|"
        r"من أنت؟?|من انت؟?|ماذا تستطيع( أن تفعل)?؟?|ماذا يمكنك( أن تفعل)?؟?|"
        r"شو بتقدر تسوي؟?|كيف (يمكنك )?(أن )?تساعدني؟?|ما هذا؟?|مساعدة)"
        + _PUNCT + r"$",
        re.IGNORECASE,
    ), 60),
    # -- about the stored user profile / memory --
    (re.compile(
        r"^(what('s| is) in your (chat )?memory\??|"
        r"what do you (know|remember)( about me)?\??|"
        r"what('s| is) my profile\??|do you (know|remember) me\??|"
        r"ماذا تعرف عني؟?|ماذا تتذكر( عني)?؟?)"
        + _PUNCT + r"$",
        re.IGNORECASE,
    ), 60),
]


def _skip_retrieval(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    return any(
        len(t) <= max_len and pattern.match(t)
        for pattern, max_len in _SKIP_RETRIEVAL_PATTERNS
    )



class RouteDecision(BaseModel):
    needs_retrieval: bool = Field(
        description="true if answering requires searching the UAE visa/residency "
        "knowledge base; false if the message can be answered without it."
    )
    search_query: str | None = Field(
        default=None,
        description="When needs_retrieval is true: a short, self-contained "
        "search query for the knowledge base, in the user's language. Resolve "
        "pronouns and references using the previous answer.",
    )
    shares_personal_facts: bool = Field(
        default=False,
        description="true when the LATEST USER MESSAGE states durable personal "
        "details about the user (nationality, profession, salary, deposits, "
        "assets, education, family status). Questions alone are false.",
    )
    erase_memory: bool = Field(
        default=False,
        description="true ONLY when the user explicitly asks to delete, clear, "
        "or forget their stored personal information. Asking what is "
        "remembered is false.",
    )


ROUTER_PROMPT = (
    "You route messages for a UAE visa/residency assistant. Decide whether the "
    "LATEST USER MESSAGE requires searching the visa-services knowledge base.\n"
    "needs_retrieval=false for: greetings/small talk; questions about the "
    "assistant itself; questions about the user's saved details; requests to "
    "transform the PREVIOUS ANSWER — translate it, repeat it, summarize it, "
    "shorten/expand it, reformat it, or explain part of it.\n"
    "needs_retrieval=true for: ANY question about visa/residency/permit "
    "services, fees, documents, eligibility, or processes — including "
    "follow-ups that ask about a different service, option, duration, or "
    "detail than the previous answer covered.\n"
    "If in doubt, choose true.\n"
    "When needs_retrieval=true, also write search_query: a concise standalone "
    "query naming the service and the detail asked about, resolving references "
    "via the previous answer (e.g. after a transit-visa answer, 'what about "
    "tourist visa' -> 'tourist visa fees'; 'what about the multi entry one?' "
    "-> 'multi entry tourist visa fees').\n"
    "Separately, set shares_personal_facts=true when the latest message states "
    "personal details or circumstances about the user themselves — their "
    "current visa/residency situation even if temporary ('I have a 30-day "
    "tourist visa' -> true), nationality, profession, salary, deposits, assets, "
    "education, family — INCLUDING when stated inside a question ('I am "
    "Jordanian, can I enter?' -> true). Merely asking about a topic does NOT "
    "count, and neither does asking what you remember ('what do you know about "
    "me?' -> false: recall request, not a disclosure).\n"
    "Separately, set erase_memory=true ONLY when the user explicitly asks to "
    "delete, clear, reset, or forget their stored personal details (e.g. "
    "'forget my information', 'delete my data'). Asking WHAT you remember is "
    "not an erase request."
)


def _router_view(state: ClarifierState, latest: str) -> list:
    prev_answer = ""
    for msg in reversed(state["messages"][:-1]):
        if isinstance(msg, AIMessage) and msg.content:
            prev_answer = str(msg.content)[:600]
            break
    return [
        HumanMessage(
            content=(
                f"PREVIOUS ANSWER (truncated):\n{prev_answer or '(none)'}\n\n"
                f"LATEST USER MESSAGE: {latest}"
            )
        )
    ]


def _needs_retrieval(state: ClarifierState, query: str) -> tuple[bool, str, bool, bool]:
    """Returns (needs_retrieval, search_query, shares_facts, erase_memory).

    Fails open to (True, verbatim, True, False): on router errors we'd rather
    waste a retrieval than lose an answer or a fact — but never erase by
    accident.
    """
    t0 = time.perf_counter()
    search_query = query
    shares_facts = True
    erase = False
    try:
        decision = clarifier_llm.with_structured_output(RouteDecision).invoke(
            [SystemMessage(content=ROUTER_PROMPT + NO_THINK)] + _router_view(state, query)
        )
        needs = decision.needs_retrieval
        shares_facts = decision.shares_personal_facts
        erase = decision.erase_memory
        if needs and decision.search_query and decision.search_query.strip():
            search_query = decision.search_query.strip()
    except Exception as e:
        logger.error(f"Error in retrieval router, defaulting to retrieve: {e}")
        needs = True
    logger.info(
        f"[timing] router: {time.perf_counter() - t0:.2f}s | retrieve={needs}"
        + (f" | query={search_query!r}" if needs else "")
        + f" | facts={shares_facts}"
        + (" | ERASE" if erase else "")
    )
    return needs, search_query, shares_facts, erase


def _retrieve_call(query: str) -> AIMessage:
    """The synthetic tool-call message that triggers the retriever node."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "retrieve",
            "args": {"query": query},
            "id": f"retrieve-{uuid4().hex[:8]}",
            "type": "tool_call",
        }],
    )


def _erase_facts(store: BaseStore) -> int:
    """Deterministic PII wipe — the LLM only flags the request, code executes it."""
    items = store.search(FACTS_NAMESPACE, limit=1000)
    for item in items:
        store.delete(FACTS_NAMESPACE, item.key)
    return len(items)


def issue_retrieve(state: ClarifierState, *, store: BaseStore = None) -> Command[Literal["retriever", "responder agent", "memory writer"]]:
    """Issue the retrieval tool call without an LLM — or skip it for small talk.

    First pass searches with the user's own words; on requery it uses the
    clarifier's improved_query directly. Pure greetings/thanks skip retrieval
    entirely (straight to the responder); the check fails toward retrieval, so
    anything with real content always searches.
    """
    query = state.get("improved_query", "")
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break
        # if _skip_retrieval(query):
        #     logger.info("[router] small talk / meta question -> skipping retrieval")
        #     return Command(goto="responder agent")
        needs, query, shares_facts, erase = _needs_retrieval(state, query)
        update = {}
        if erase and store is not None:
            try:
                n = _erase_facts(store)
                update["memory_erased"] = True
                shares_facts = False  # a wipe turn never re-saves
                logger.info(f"[memory] erased {n} facts by user request")
            except Exception as e:
                logger.error(f"Error erasing memory: {e}")
        # early dispatch: the memory writer runs as a PARALLEL branch alongside
        # retrieval/responding — durable before the run ends, but off the
        # critical path instead of appended after the responder
        targets = []
        if needs:
            update["messages"] = [_retrieve_call(query)]
            targets.append("retriever")
        else:
            logger.info("[router] LLM router -> skipping retrieval")
            targets.append("responder agent")
        if shares_facts and store is not None:
            targets.append("memory writer")
        return Command(goto=targets, update=update)
    # requery pass: routing was decided on the first pass this turn
    return Command(goto="retriever", update={"messages": [_retrieve_call(query)]})


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


def clarifier_agent(state: ClarifierState) -> Command[Literal["issue retrieve", "responder agent"]]:
    """Judges the query/retrieval and routes: proceed, requery, or reprompt the user."""
    count = state.get("clarify_count", 0)
    try:
        t0 = time.perf_counter()

        # Loop guard: after MAX_CLARIFY cycles, stop clarifying and just answer.
        if count >= MAX_CLARIFY:
            logger.info(f"[clarifier] cap reached ({count}) -> proceed")
            return Command(goto="responder agent")

        system = SystemMessage(content=CLARIFIER_PROMPT + NO_THINK)
        decision = clarifier_llm.with_structured_output(ClarifierDecision).invoke([system] + _clarifier_view(state))
        logger.info(
            f"[timing] clarifier node: {time.perf_counter() - t0:.2f}s "
            f"| action={decision.action} | count={count}"
        )

        if decision.action == "requery" and decision.improved_query:
            # Re-run retrieval with the clarifier's own query — no LLM rewrite,
            # no synthetic user message polluting the thread/UI.
            logger.info(f"[clarifier] requery -> {decision.improved_query!r}")
            return Command(
                goto="issue retrieve",
                update={
                    "improved_query": decision.improved_query,
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


def _recall_facts(store: BaseStore | None, state: ClarifierState) -> str | None:
    """Semantic recall: the top facts relevant to the current question."""
    if store is None:
        return None
    try:
        t0 = time.perf_counter()
        question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break
        hits = store.search(FACTS_NAMESPACE, query=question or None, limit=5)
        logger.info(
            f"[timing] memory recall: {time.perf_counter() - t0:.2f}s | hits={len(hits)}"
        )
        if not hits:
            return None
        return "\n".join(f"- {h.value.get('text', '')}" for h in hits)
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        return None


def memory_recall(state: ClarifierState, *, store: BaseStore = None):
    """Parallel branch off `reset`: fetches relevant facts concurrently with the
    router, so recall costs nothing on the responder's critical path."""
    return {"recalled_facts": _recall_facts(store, state) or ""}


def responder_agent(state: ClarifierState):
    """Streams the final answer, OR (mode='ask') a clarifying question back to the user."""
    try:
        t0 = time.perf_counter()
        mode = state.get("mode", "answer")
        # recall ran in parallel at turn start; if this turn erased memory,
        # that snapshot predates the wipe — don't inject it
        facts = None if state.get("memory_erased") else (state.get("recalled_facts") or None)
        prompt = build_system_prompt(facts)
        if state.get("memory_erased"):
            prompt += (
                "\n\nNOTE: The user's stored personal details were just erased at "
                "their request. Confirm clearly that everything was deleted."
            )
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


# ---- long-term memory writer: runs AFTER the answer has streamed ----
class MemoryFact(BaseModel):
    key: str = Field(
        description="stable snake_case id for the fact type, e.g. 'nationality', "
        "'bank_deposit_aed', 'profession' — newer values overwrite older ones"
    )
    text: str = Field(
        description="short third-person fact, e.g. 'User is an Indian national'"
    )


class MemoryExtraction(BaseModel):
    facts: list[MemoryFact] = Field(
        default_factory=list,
        description="durable personal facts the USER stated; empty if none",
    )


MEMORY_WRITER_PROMPT = (
    "You maintain long-term memory for a UAE visa assistant. From the exchange, "
    "extract durable personal facts the USER stated about themselves that matter "
    "for visa services or personalization: current visa/residency status and its "
    "type, duration or expiry (e.g. 'has a 30-day tourist visa'); whether they "
    "are inside or outside the UAE; sponsor or employer; nationality, age, "
    "profession, status (investor/student/employee), salary, bank deposit, "
    "property value, company investment, education, GPA, family situation — or "
    "any other durable detail a visa officer would note. Rules:\n"
    "- Only facts the USER stated about themselves in USER SAID. The previous "
    "assistant message is context for resolving references (e.g. user answers "
    "'2 million' to a question about deposits) — never a fact source itself. "
    "Never store the question topic itself.\n"
    "- Use a stable snake_case key per fact type (e.g. 'salary_aed', 'nationality') "
    "so newer values overwrite older ones.\n"
    "- If no durable personal facts were shared, return an empty list."
)


def memory_writer(state: ClarifierState, *, store: BaseStore = None):
    """Extracts durable user facts into the store, dispatched by the router as
    a parallel branch (it never blocks the responder, which doesn't read it).

    Durable-by-design: the run only reports finished once the facts are
    written — but since this runs alongside retrieval/responding, that
    guarantee no longer costs tail latency. Extraction reads the USER's
    message only; the previous answer is context for resolving references
    (this turn's answer doesn't exist yet — by construction it can never
    contaminate extraction)."""
    if store is None:
        return {}
    try:
        t0 = time.perf_counter()
        question, prev_answer = "", ""
        for msg in reversed(state["messages"]):
            if not prev_answer and isinstance(msg, AIMessage) and msg.content:
                prev_answer = str(msg.content)[:400]
            elif isinstance(msg, HumanMessage):
                question = str(msg.content)
                break
        if not question:
            return {}
        view = HumanMessage(
            content=(
                f"PREVIOUS ASSISTANT MESSAGE (context only):\n{prev_answer or '(none)'}\n\n"
                f"USER SAID: {question}"
            )
        )
        result = clarifier_llm.with_structured_output(MemoryExtraction).invoke(
            [SystemMessage(content=MEMORY_WRITER_PROMPT + NO_THINK), view]
        )
        for f in result.facts:
            store.put(FACTS_NAMESPACE, f.key, {"text": f.text}, index=["text"])
        logger.info(
            f"[timing] memory writer: {time.perf_counter() - t0:.2f}s "
            f"| facts={[f.key for f in result.facts]}"
        )
    except Exception as e:
        logger.error(f"Error in memory writer: {e}")
    return {}


builder = StateGraph(ClarifierState)
builder.add_node("reset", reset_turn)
builder.add_node("memory recall", memory_recall)
builder.add_node("issue retrieve", issue_retrieve)
builder.add_node("retriever", ToolNode(TOOLS))
builder.add_node("clarifier", clarifier_agent)
builder.add_node("responder agent", responder_agent)
builder.add_node("memory writer", memory_writer)

builder.add_edge(START, "reset")
builder.add_edge("reset", "issue retrieve")
builder.add_edge("reset", "memory recall")  # parallel branch: hides recall
# latency under the router LLM call; the superstep barrier joins them
# "issue retrieve" routes itself via Command: -> "retriever" (normal questions,
# and requeries with the clarifier's improved_query) or -> "responder agent"
# (pure small talk, no retrieval).
builder.add_edge("retriever", "clarifier")
# "clarifier" routes itself via Command(goto=...) -> "issue retrieve" (requery) or "responder agent"
# "memory writer" has no static edges: the router dispatches it in parallel
# via Command(goto=[...]) on fact-bearing turns; it dead-ends when done.
builder.add_edge("responder agent", END)

# Compiled with an in-memory checkpointer; main.py's startup hook replaces it
# with AsyncPostgresSaver (needs a running event loop) for durable conversations.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, store=memory_store)

# Single local user
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
