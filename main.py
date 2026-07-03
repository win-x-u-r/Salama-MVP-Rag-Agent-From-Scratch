import os
import json
import logging
import time
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangGraph agent graph + short-term (conversation) memory ---
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
logger = logging.getLogger("uvicorn.error")
#setting up the llm
try:
    logger.info("Initializing LLM...")
    llm = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0, #(it's cold in Dubai, we don't want the model to hallucinate)
        reasoning={"effort": "none"},  # disable chain-of-thought reasoning for faster responses
        openrouter_provider={"sort": "throughput"}, # prefer the fastest qwen3 provider
        seed=42,  # the answer of life
    )
    logger.info("LLM initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")


#splitter
try:
    def _load_chunks() -> list[Document]:
        logger.info("Loading and splitting documents...")
        # docs = [
        #     Document(page_content=open("rag-file.txt", encoding="utf-8").read(), metadata={"source": "rag-file.txt"})
        # ]
        # splitted_docs = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50).split_documents(docs)
        with open("./gdrfa_records.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info(f"Loaded JSON data with {len(data)} records")
        splitted_docs = [
            Document(
                page_content=json.dumps(record, ensure_ascii=False),
                metadata={"record_index": i},
            )
            for i, record in enumerate(data)
        ]
        logger.info(f"Documents loaded and split successfully: {len(splitted_docs)} chunks created")
        return splitted_docs
except Exception as e:
    logger.error(f"Error loading or splitting documents: {e}")

#retriever
# try:
#     logger.info("Initializing retriever...")
#     retriever = InMemoryVectorStore.from_documents(
#         _load_chunks(), FastEmbedEmbeddings()
#     ).as_retriever(search_kwargs={"k": 4})
#     fast_embed = FastEmbedEmbeddings.aembed_documents
#     logger.info("Retriever initialized successfully.")
# except Exception as e:
#     logger.error(f"Error initializing retriever: {e}")

#embedding model setup
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    # default_headers={
    #     "HTTP-Referer": "http://localhost:8000", # OpenRouter requires a referer
    #     "X-Title": "Agentic RAG App"
    # }
)# some workaround to route the openai embedding model through openrouter

# ======================================================================
# Retrieval backend: "qdrant" (vector-only) or "graph" (Neo4j hybrid).
# Set RETRIEVER_BACKEND in .env to switch. Defaults to qdrant.
# ======================================================================
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "qdrant").lower()

graph_retriever = None
if RETRIEVER_BACKEND == "graph":
    try:
        logger.info("setting up Neo4j graph retriever")
        from graph_retriever import GraphRetriever

        graph_retriever = GraphRetriever(embeddings)
        logger.info("Neo4j graph retriever ready.")
    except Exception as e:
        logger.error(f"Error setting up Neo4j graph retriever: {e}")

#Qdrant setup
qdrant = None
try:
    if RETRIEVER_BACKEND == "graph":
        raise RuntimeError("skipped (RETRIEVER_BACKEND=graph)")
    # use existing collection:
    logger.info("setting up Qdrant client")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="rag_collection",
        path=("./qdrant_collection"),
    )

    # # make new collection:
    # logger.info("setting up Qdrant client")
    # client = QdrantClient(path="./qdrant_collection")
    # client.create_collection(
    #     collection_name="rag_collection",
    #     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    # )
    # logger.info("setting vector store")
    # qdrant = QdrantVectorStore(
    # client=client,
    # collection_name="rag_collection",
    # embedding=embeddings,
    # )
    # logger.info("loading chunks")
    # chunked_documents = _load_chunks()
    # logger.info("setting uuids")
    # uuids = [str(uuid4()) for _ in range(len(chunked_documents))]
    # logger.info("adding documents to qdrant")
    # qdrant.add_documents(documents=chunked_documents, ids=uuids)
    # logger.info("Qdrant client and collection set up successfully.")

except Exception as e:
    logger.error(f"Error setting up Qdrant client and collection: {e}")


#retrieving tool
@tool
def retrieve(query: str, k: int = 3) -> str:
    """Search the knowledge base and return relevant passages.

    Args:
        query: What to search for.
        k: How many passages to return. Use a LARGER value (6-10) for broad,
           definitional, or "list all"-style questions (e.g. "what is the golden
           visa", "what visa types exist") since the answer may span several
           services. Use a SMALL value (2-4) for narrow factual lookups
           (a single fee, date, GPA, or document requirement).
    """
    try:
        requested_k = k
        k = max(1, min(k, 10))  # clamp so the model can't ask for too few/many
        clamp_note = "" if k == requested_k else f" (clamped from {requested_k})"
        logger.info(
            f"Retrieving documents for query: {query} | k={k}{clamp_note} "
            f"| backend={RETRIEVER_BACKEND}"
        )
        t0 = time.perf_counter()
        if RETRIEVER_BACKEND == "graph" and graph_retriever is not None:
            results = graph_retriever.retrieve(query, k=k)
        else:
            query_vector = embeddings.embed_query(query)
            docs = qdrant.similarity_search_by_vector(query_vector, k=k)
            results = "\n\n".join(d.page_content for d in docs)
        logger.info(f"[timing] retrieve total={time.perf_counter() - t0:.2f}s")
        logger.info(f"Documents retrieved successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return "An error occurred while retrieving documents."


# long term memory (for personal context)
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")


def load_profile() -> dict:
    """Read the saved user profile (returns {} if none yet)."""
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_profile(updates: dict) -> dict:
    """Merge `updates` into the stored profile and write it back to disk."""
    profile = load_profile()
    # ignore empty values so the model can't wipe fields with blanks
    profile.update({k: v for k, v in updates.items() if v not in (None, "")})
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


@tool
def save_user_profile(updates: dict) -> str:
    """Save or update the user's personal details for visa-eligibility checks.

    Call this whenever the user shares personal information relevant to visa
    eligibility, such as: nationality, age, profession, status (investor /
    student / employee / self-employed), salary, bank deposit amount, property
    value, company investment, annual tax paid, GPA, university, education level.

    Pass a flat object of field -> value, for example:
    {"nationality": "Indian", "status": "investor", "bank_deposit_aed": 2000000}

    Only include fields the user actually provided. Existing fields are kept.
    """
    try:
        profile = _save_profile(updates)
        logger.info(f"[profile] updated -> {profile}")
        return f"Saved. Current profile: {json.dumps(profile, ensure_ascii=False)}"
    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        return "Could not save the profile."


@tool
def clear_user_profile() -> str:
    """Erase ALL saved personal details about the user.

    Call this only when the user explicitly asks to clear, delete, reset, or
    forget their stored information (e.g. "forget my details", "delete my data").
    This cannot be undone.
    """
    try:
        if os.path.exists(PROFILE_PATH):
            os.remove(PROFILE_PATH)
        logger.info("[profile] cleared by user request")
        return "Done. All saved personal details have been erased."
    except Exception as e:
        logger.error(f"Error clearing user profile: {e}")
        return "Could not clear the profile."


# inject user profile in system prompt
BASE_SYSTEM_PROMPT = (
    "You are the GDRFA Dubai visa assistant. You answer questions about UAE visa and "
    "residency services by searching a knowledge base with the `retrieve` tool.\n\n"
    "ANSWERING:\n"
    "- ALWAYS Use `retrieve` for any visa/residency question. Pick k=2-3 for narrow factual "
    "lookups and k=6-8 for broad or 'what is / what types' questions.\n"
    "- Scope answers to exactly what is asked; for a specific question do not list the "
    "other options found in the context.\n"
    "- For cost/fee questions: give the main fee, then the always-applicable mandatory "
    "fees (Knowledge Dirham, Innovation Dirham, VAT) and an approximate total.\n"
    "- Be concise: short sentences or bullet points in maximum 50 words. If the answer isn't in the knowledge "
    "base, say so. If the question is unrelated to UAE visas, say so without retrieving.\n\n"
    "PERSONAL DETAILS & ELIGIBILITY:\n"
    "- When the user shares personal details relevant to eligibility (nationality, age, "
    "profession, status, salary, bank deposit, property value, company investment, annual "
    "tax, GPA, university, education), call `save_user_profile` to store them.\n"
    "- For eligibility questions ('am I eligible for X'): FIRST `retrieve` that visa's "
    "requirements, THEN compare them against the user's KNOWN PROFILE below.\n"
    "- If a required detail is missing from the profile, ASK the user for that specific "
    "detail instead of guessing. When they answer, save it and continue.\n"
    "- Give a clear verdict: which requirements are met, which are not, and the bottom line.\n"

)


def build_system_prompt() -> str:
    """Base instructions + the live user profile, injected on every turn."""
    profile = load_profile()
    profile_text = (
        json.dumps(profile, ensure_ascii=False, indent=2)
        if profile
        else "(no personal details saved yet)"
    )
    return f"{BASE_SYSTEM_PROMPT}\n\nKNOWN USER PROFILE:\n{profile_text}"


# ======================================================================
# The LangGraph agent graph (ReAct loop): agent <-> tools
#
#   START -> agent -> (tools_condition) -> tools -> agent -> ... -> END
#
# The single "agent" node both decides to call a tool and writes the final
# answer once tool results come back. The profile is injected fresh into the
# system prompt each turn; it is NOT stored in state, so it never goes stale.
# ======================================================================
TOOLS = [retrieve, save_user_profile, clear_user_profile]


def agent_node(state: MessagesState):
    """Call the LLM with the tools + a freshly-built (profile-aware) system prompt."""
    try:
        t0 = time.perf_counter()
        system = SystemMessage(content=build_system_prompt())
        response = llm.bind_tools(TOOLS).invoke([system] + state["messages"])
        logger.info(f"[timing] agent node (LLM): {time.perf_counter() - t0:.2f}s")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent node: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred while processing your request."}]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # -> "tools" if a tool was called, else END
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()  # short-term (conversation) memory, per thread
graph = builder.compile(checkpointer=checkpointer)

# Single local user -> one persistent conversation thread.
THREAD_ID = "local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}


#fastapi stuff
app = FastAPI(title="Agentic RAG")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

class Query(BaseModel):
    question: str


@app.get("/profile")
def get_profile():
    """Inspect what the agent currently remembers about the user."""
    return load_profile()


@app.post("/profile/reset")
def reset_profile():
    """Clear the saved personal details."""
    try:
        if os.path.exists(PROFILE_PATH):
            os.remove(PROFILE_PATH)
        return {"status": "profile cleared"}
    except Exception as e:
        logger.error(f"Error resetting profile: {e}")
        return {"error": "could not reset profile"}


@app.post("/chat")
def chat(q: Query):
    try:
        logger.info("Processing chat request for question")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": q.question}]},
            config=GRAPH_CONFIG,
        )
        logger.info(f"Finished processing chat request for question, response: {result['messages'][-1].content}")
        return {"answer": result["messages"][-1].content}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return {"error": "An error occurred while processing your request."}


@app.post("/chat/stream")
def chat_stream(q: Query):
    def token_gen():
        logger.info("Processing streaming chat request for question")
        req_start = time.perf_counter()
        first_token_logged = False
        try:
            for token, metadata in graph.stream(
                {"messages": [{"role": "user", "content": q.question}]},
                stream_mode="messages",
                config=GRAPH_CONFIG,
            ):
                if metadata.get("langgraph_node") == "agent" and token.content:
                    if not first_token_logged:
                        logger.info(f"[timing] time-to-first-token: {time.perf_counter() - req_start:.2f}s")
                        first_token_logged = True
                    yield token.content
            logger.info(f"[timing] total request: {time.perf_counter() - req_start:.2f}s")
            logger.info("Finished streaming chat request for question")
        except Exception as e:
            logger.error(f"Error processing streaming chat request: {e}")
            yield "\n[An error occurred while processing your request.]"

    return StreamingResponse(
        token_gen(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},  # disable proxy buffering
    )
