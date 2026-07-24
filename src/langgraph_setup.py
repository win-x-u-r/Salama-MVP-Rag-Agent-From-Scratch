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
from .service_panel import (
    service_panel,
    panel_will_render,
    PANEL_TOOL_NAME,
    COMPARE_TOOL_NAME,
)
from .dynamic_ui import DYNAMIC_TOOL_NAME

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
    save_facts: bool     # router flagged personal facts -> memory writer runs
    compare_services: bool  # router flagged a comparative question -> panel node
                            # may emit a two-service comparison chart
    compare_targets: list   # the two compared services as search phrases


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
        "save_facts": False,
        "compare_services": False,
        "compare_targets": [],
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
    # RECALL-BIASED GATE (contract): this flag only decides whether the memory
    # writer runs; the extractor+reconciler own precision. A false positive
    # costs seconds of post-stream compute, a false negative loses a fact —
    # so this must over-trigger, never under-trigger.
    shares_personal_facts: bool = Field(
        default=False,
        description="true when the LATEST USER MESSAGE contains anything worth "
        "remembering across conversations: information about the user OR "
        "standing preferences/instructions for the assistant. Test: would "
        "future conversations go differently for knowing this? Unsure -> true.",
    )
    erase_memory: bool = Field(
        default=False,
        description="true ONLY when the user explicitly asks to delete, clear, "
        "or forget their stored personal information. Asking what is "
        "remembered is false.",
    )
    compare_services: bool = Field(
        default=False,
        description="true when the LATEST USER MESSAGE asks to COMPARE two "
        "different visa/residency services against each other (difference "
        "between X and Y, X versus Y, which of X or Y is cheaper/faster). "
        "Asking about a single service, or about options WITHIN one service "
        "(e.g. 30 vs 60 days of the same visa), is false.",
    )
    compare_targets: list[str] = Field(
        default_factory=list,
        description="when compare_services is true: the TWO services being "
        "compared, each as its own short standalone search phrase (e.g. "
        "['golden visa for investors', 'green visa self-employment']). "
        "Empty otherwise.",
    )


ROUTER_PROMPT = (
    "You route messages for a UAE visa/residency assistant. Decide whether the "
    "LATEST USER MESSAGE requires searching the visa-services knowledge base.\n"
    "needs_retrieval=false for: greetings/small talk; questions about the "
    "assistant itself; questions about the user's saved details; requests to "
    "transform the PREVIOUS ANSWER — translate it, repeat it, summarize it, "
    "shorten/expand it, or explain part of it.\n"
    "EXCEPTION — visualization requests ALWAYS retrieve: asking for a chart, "
    "pie/donut, graph, table, breakdown, or comparison of service data "
    "(even 'show that as a pie chart' about the previous answer) is "
    "needs_retrieval=true, with search_query naming the underlying service "
    "resolved from the previous answer (e.g. '5-year tourist visa fees').\n"
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
    "Separately, set shares_personal_facts=true when the message contains "
    "ANYTHING worth remembering across conversations. There are no "
    "categories; apply one test: would the assistant behave differently in a "
    "FUTURE conversation for having known this? Two broad types pass it:\n"
    "(1) information about the user or their circumstances — even inside a "
    "question ('I am Jordanian, can I enter?' -> true; 'I have a 30-day "
    "tourist visa' -> true);\n"
    "(2) standing preferences or instructions for how the assistant should "
    "behave ('talk to me in Arabic from now on' -> true; 'keep answers "
    "short' -> true).\n"
    "When unsure, choose true — a wasted check is cheap, a lost memory is "
    "not. Only two things never count: merely asking about a topic, and "
    "asking what you remember ('what do you know about me?' -> false).\n"
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


def _needs_retrieval(state: ClarifierState, query: str):
    """Returns (needs_retrieval, search_query, shares_facts, erase_memory,
    compare, compare_targets).

    Fails open to (True, verbatim, True, False, False, []): on router errors
    we'd rather waste a retrieval than lose an answer or a fact — but never
    erase by accident, and a missed comparison just falls back to a single
    panel.
    """
    t0 = time.perf_counter()
    search_query = query
    shares_facts = True
    erase = False
    compare = False
    targets: list[str] = []
    try:
        decision = llm.with_structured_output(RouteDecision).invoke(
            [SystemMessage(content=ROUTER_PROMPT + NO_THINK)] + _router_view(state, query)
        )
        needs = decision.needs_retrieval
        shares_facts = decision.shares_personal_facts
        erase = decision.erase_memory
        compare = decision.compare_services
        targets = [t.strip() for t in (decision.compare_targets or []) if t and t.strip()]
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
        + (f" | COMPARE {targets}" if compare else "")
    )
    return needs, search_query, shares_facts, erase, compare, targets


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


def issue_retrieve(state: ClarifierState, *, store: BaseStore = None) -> Command[Literal["retriever", "responder agent"]]:
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
        needs, query, shares_facts, erase, compare, targets = _needs_retrieval(state, query)
        update = {"compare_services": compare, "compare_targets": targets}
        if erase and store is not None:
            try:
                n = _erase_facts(store)
                update["memory_erased"] = True
                shares_facts = False  # a wipe turn never re-saves
                logger.info(f"[memory] erased {n} facts by user request")
            except Exception as e:
                logger.error(f"Error erasing memory: {e}")
        # the facts flag gates the post-response memory writer
        update["save_facts"] = shares_facts
        if needs:
            update["messages"] = [_retrieve_call(query)]
            return Command(goto="retriever", update=update)
        logger.info("[router] LLM router -> skipping retrieval")
        return Command(goto="responder agent", update=update)
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
    """Two-lane recall.

    Lane 1 — standing preferences: ALWAYS injected, no semantic matching
    (they're relevant to every turn but semantically near no query). A plain
    filtered lookup, no embedding call.
    Lane 2 — facts: semantic top-K against the current question, as before.
    """
    if store is None:
        return None
    try:
        t0 = time.perf_counter()
        question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = str(msg.content)
                break
        prefs = store.search(FACTS_NAMESPACE, filter={"kind": "preference"}, limit=20)
        hits = store.search(FACTS_NAMESPACE, query=question or None, limit=5)
        pref_keys = {p.key for p in prefs}
        fact_hits = [h for h in hits if h.key not in pref_keys]
        logger.info(
            f"[timing] memory recall: {time.perf_counter() - t0:.2f}s "
            f"| prefs={len(prefs)} facts={len(fact_hits)}"
        )
        if not prefs and not fact_hits:
            return None
        sections = []
        if prefs:
            sections.append(
                "STANDING PREFERENCES (always follow):\n"
                + "\n".join(f"- {p.value.get('text', '')}" for p in prefs)
            )
        if fact_hits:
            sections.append(
                "FACTS:\n" + "\n".join(f"- {h.value.get('text', '')}" for h in fact_hits)
            )
        return "\n\n".join(sections)
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        return None


def memory_recall(state: ClarifierState, *, store: BaseStore = None):
    """Parallel branch off `reset`: fetches relevant facts concurrently with the
    router, so recall costs nothing on the responder's critical path."""
    return {"recalled_facts": _recall_facts(store, state) or ""}


_UI_TOOL_NAMES = {PANEL_TOOL_NAME, COMPARE_TOOL_NAME, DYNAMIC_TOOL_NAME}


def _llm_view(messages: list) -> list:
    """History minus panel/comparison tool-call pairs — they are UI artifacts
    assembled from the same passages the LLM already sees; replaying their JSON
    payloads into the context every turn is pure token cost."""
    out = []
    for m in messages:
        if (
            isinstance(m, AIMessage)
            and m.tool_calls
            and all(tc["name"] in _UI_TOOL_NAMES for tc in m.tool_calls)
        ):
            continue
        if isinstance(m, ToolMessage) and getattr(m, "name", None) in _UI_TOOL_NAMES:
            continue
        out.append(m)
    return out


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
        elif panel_will_render(state):
            # The panel/chart repeats the detailed data — prose that enumerates
            # it again is the main source of overlong answers (measured avg 59
            # words, comparisons 90-134). Defer details to the panel.
            prompt += (
                "\n\nIMPORTANT — A DATA PANEL IS ATTACHED: a visual panel/chart "
                "below your answer displays this service's detailed fees, "
                "documents, eligibility and totals. Do NOT enumerate them in "
                "prose. Reply in 1-2 short sentences (max 35 words): the direct "
                "answer or headline figure, at most one caveat, then point to "
                "the details below (e.g. 'Full breakdown below.'). No bullet "
                "lists. No closing offers."
            )
        payload = [SystemMessage(content=prompt)] + _llm_view(state["messages"])
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


# ---- long-term memory writer: extract -> neighbor search -> reconcile ----
# Runs after the responder. Memories are FREEFORM sentences (no schema, no
# category list); consistency comes from the reconciler, not from keys.
class ExtractedMemory(BaseModel):
    text: str = Field(
        description="short standalone third-person sentence, e.g. 'User has "
        "3 brothers' or 'User wants the assistant to speak Arabic'"
    )
    kind: Literal["fact", "preference"] = Field(
        description="'preference' = a standing instruction for how the "
        "assistant should behave (language, format, style, how to address "
        "the user); 'fact' = information about the user or their circumstances"
    )


class ExtractedFacts(BaseModel):
    facts: list[ExtractedMemory] = Field(
        default_factory=list,
        description="memories worth keeping; empty if nothing lasting was shared",
    )


EXTRACT_PROMPT = (
    "You extract long-term memory for a UAE visa assistant from one exchange.\n"
    "Source: ONLY what the USER said. Assistant messages are context for "
    "resolving references (e.g. the user answers '2 million' to the "
    "assistant's question about deposits) — never a fact source.\n"
    "Extract ANYTHING worth remembering across conversations, as short "
    "standalone third-person sentences. There are no categories; apply one "
    "test: would a good assistant behave differently in a FUTURE conversation "
    "for having known this? Two broad types pass it:\n"
    "- information about the user or their circumstances, e.g. 'User has 3 "
    "brothers'\n"
    "- standing preferences or instructions for the assistant, e.g. 'User "
    "wants the assistant to speak Arabic from now on', 'User prefers short "
    "answers'\n"
    "Also include logically ENTAILED derivations, with the reasoning in "
    "parentheses, e.g. 'User has 5 siblings (3 brothers + 2 sisters)'. Only "
    "strict entailment (arithmetic, category rollups) — never plausible "
    "guesses.\n"
    "Skip what fails the test: small talk, one-off questions, and requests "
    "that only apply to the current turn ('translate THIS to Arabic' is "
    "one-off; 'speak Arabic from now on' is standing).\n"
    "State each distinct memory exactly ONCE — never restate the same one in "
    "different words.\n"
    "Tag each memory's kind: 'preference' for standing instructions about how "
    "the assistant should behave; 'fact' for everything else.\n"
    "Return an empty list if nothing passes the test."
)


class ReconcileDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "NOOP"] = Field(
        description="NOOP = already stored; UPDATE = an existing memory is "
        "contradicted or refined; ADD = genuinely new information"
    )
    id: str | None = Field(
        default=None, description="the existing memory id to rewrite (UPDATE only)"
    )
    text: str | None = Field(
        default=None, description="the memory sentence to store (ADD and UPDATE)"
    )


class ReconcileDecisions(BaseModel):
    decisions: list[ReconcileDecision] = Field(default_factory=list)


RECONCILE_PROMPT = (
    "You maintain a user's memory store. For each NEW fact you are given its "
    "most similar EXISTING memories (with ids). Decide ONE action per new "
    "fact, in the same order.\n"
    "Rule: each ATTRIBUTE of the user (their nationality, their salary, their "
    "sibling count, their visa status, ...) must end up with exactly ONE "
    "memory row.\n"
    "- The attribute already has a row with the SAME meaning ('is from "
    "Jordan' = 'is Jordanian') -> NOOP.\n"
    "- The attribute already has a row with a DIFFERENT value -> UPDATE: "
    "return that row's id plus the corrected sentence.\n"
    "- The attribute has no row at all -> ADD: return the sentence. "
    "Unrelated existing memories are never a reason to skip an ADD.\n"
    "An earlier NEW fact in this batch counts as that attribute's row for "
    "the facts after it.\n"
    "Exception: a DERIVED/aggregate fact (one with reasoning in parentheses, "
    "e.g. 'User has 5 siblings (3 brothers + 2 sisters)') is its own "
    "attribute — never NOOP it against the component facts it derives from."
)


def _extract(question: str, answer: str, prev_answer: str) -> list[str]:
    """Step 1: freeform fact extraction from this turn's exchange."""
    view = HumanMessage(
        content=(
            f"PREVIOUS ASSISTANT MESSAGE (context only):\n{prev_answer or '(none)'}\n\n"
            f"USER SAID: {question}\n\n"
            f"ASSISTANT REPLIED (context only):\n{answer or '(none)'}"
        )
    )
    result = clarifier_llm.with_structured_output(ExtractedFacts).invoke(
        [SystemMessage(content=EXTRACT_PROMPT + NO_THINK), view]
    )
    return [f for f in result.facts if f.text and f.text.strip()]


def _reconcile(facts: list[ExtractedMemory], store: BaseStore) -> list[str]:
    """Steps 2+3: neighbor search per fact, one reconciler call, apply.

    Returns a summary of applied actions for logging. Idempotent by design:
    reprocessing the same facts reconciles to NOOP. The memory's kind
    (fact/preference) travels with it through ADD and UPDATE so preferences
    never silently degrade into facts.
    """
    blocks = []
    for i, f in enumerate(facts):
        neighbors = store.search(FACTS_NAMESPACE, query=f.text, limit=4)
        existing = "\n".join(
            f"  - id={n.key} :: {n.value.get('text', '')}" for n in neighbors
        ) or "  (none)"
        blocks.append(f"NEW FACT {i + 1}: {f.text}\nEXISTING SIMILAR MEMORIES:\n{existing}")
    view = HumanMessage(content="\n\n".join(blocks))
    # the reconciler is the judgment-critical step and runs post-stream where
    # latency is cheap -> use the main model, not the small one
    result = llm.with_structured_output(ReconcileDecisions).invoke(
        [SystemMessage(content=RECONCILE_PROMPT + NO_THINK), view]
    )
    applied = []
    for i, d in enumerate(result.decisions):
        source = facts[i] if i < len(facts) else None
        text = (d.text or (source.text if source else "")).strip()
        kind = source.kind if source else "fact"
        if d.action == "ADD" and text:
            store.put(FACTS_NAMESPACE, str(uuid4()), {"text": text, "kind": kind}, index=["text"])
            applied.append(f"ADD[{kind}]:{text[:40]}")
        elif d.action == "UPDATE" and d.id and text:
            store.put(FACTS_NAMESPACE, d.id, {"text": text, "kind": kind}, index=["text"])
            applied.append(f"UPDATE[{kind}]:{text[:40]}")
        else:
            applied.append("NOOP")
    return applied


def memory_writer(state: ClarifierState, *, store: BaseStore = None):
    """Post-response memory commit: extract freeform facts from this turn's
    exchange, then reconcile them against existing memories (ADD/UPDATE/NOOP)
    so the store never accumulates duplicates or contradictions.

    The run only reports finished once memory is durably written; the answer
    has already streamed by the time this runs. Gated by the router's
    save_facts flag so ordinary questions pay nothing."""
    if store is None or not state.get("save_facts"):
        return {}
    try:
        t0 = time.perf_counter()
        question, answer, prev_answer = "", "", ""
        for msg in reversed(state["messages"]):
            if not question and isinstance(msg, AIMessage) and msg.content and not answer:
                answer = str(msg.content)[:400]
            elif isinstance(msg, HumanMessage) and not question:
                question = str(msg.content)
            elif question and isinstance(msg, AIMessage) and msg.content:
                prev_answer = str(msg.content)[:400]
                break
        if not question:
            return {}
        facts = _extract(question, answer, prev_answer)
        if not facts:
            logger.info(f"[timing] memory writer: {time.perf_counter() - t0:.2f}s | no facts")
            return {}
        applied = _reconcile(facts, store)
        logger.info(
            f"[timing] memory writer: {time.perf_counter() - t0:.2f}s | {applied}"
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
builder.add_node("service panel", service_panel)
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
# panel attaches after the answer (deterministic, no LLM), then memory commit;
# both run post-stream so neither costs visible latency
builder.add_edge("responder agent", "service panel")
builder.add_edge("service panel", "memory writer")
builder.add_edge("memory writer", END)

# Compiled with an in-memory checkpointer; main.py's startup hook replaces it
# with AsyncPostgresSaver (needs a running event loop) for durable conversations.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, store=memory_store)

# Single local user
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
