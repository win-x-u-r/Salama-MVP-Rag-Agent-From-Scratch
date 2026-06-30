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

# --- new agent stack (LangChain v1 create_agent + middleware) ---
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
)
from langgraph.checkpoint.memory import InMemorySaver

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

#Qdrant setup
try:
    # use existing collection:
    logger.info("setting up Qdrant client")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="rag_collection",
        path=("./qdrant_collection"),
    )

    # ## make new collection:
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
        k: How many passages to return. Use a LARGER value (6-8) for broad,
           definitional, or "list all"-style questions (e.g. "what is the golden
           visa", "what visa types exist") since the answer may span several
           services. Use a SMALL value (2-3) for narrow factual lookups
           (a single fee, date, GPA, or document requirement).
    """
    try:
        requested_k = k
        k = max(1, min(k, 10))  # clamp so the model can't ask for too few/many
        clamp_note = "" if k == requested_k else f" (clamped from {requested_k})"
        logger.info(f"Retrieving documents for query: {query} | k={k}{clamp_note}")
        t0 = time.perf_counter()
        query_vector = embeddings.embed_query(query)
        t_embed = time.perf_counter() - t0
        t1 = time.perf_counter()
        docs = qdrant.similarity_search_by_vector(query_vector, k=k)
        t_search = time.perf_counter() - t1
        results = "\n\n".join(d.page_content for d in docs)
        logger.info(
            f"[timing] retrieve: embed={t_embed:.2f}s search={t_search:.2f}s "
            f"total={t_embed + t_search:.2f}s"
        )
        logger.info(f"Documents retrieved successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return "An error occurred while retrieving documents."


# ======================================================================
# Long-term personal memory: a simple JSON-file user profile.
# This holds the eligibility-relevant details the user shares (nationality,
# assets, status, GPA, ...) and PERSISTS across server restarts.
# ======================================================================
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


# ======================================================================
# System prompt + live profile injection (dynamic_prompt middleware)
# ======================================================================
BASE_SYSTEM_PROMPT = (
    "You are the GDRFA Dubai visa assistant. You answer questions about UAE visa and "
    "residency services by searching a knowledge base with the `retrieve` tool.\n\n"
    "ANSWERING:\n"
    "- Use `retrieve` for any visa/residency question. Pick k=2-3 for narrow factual "
    "lookups and k=6-8 for broad or 'what is / what types' questions.\n"
    "- Scope answers to exactly what is asked; for a specific question do not list the "
    "other options found in the context.\n"
    "- For cost/fee questions: give the main fee, then the always-applicable mandatory "
    "fees (Knowledge Dirham, Innovation Dirham, VAT) and an approximate total.\n"
    "- Be concise: short sentences or bullet points. If the answer isn't in the knowledge "
    "base, say so. If the question is unrelated to UAE visas, say so without retrieving.\n\n"
    "PERSONAL DETAILS & ELIGIBILITY:\n"
    "- When the user shares personal details relevant to eligibility (nationality, age, "
    "profession, status, salary, bank deposit, property value, company investment, annual "
    "tax, GPA, university, education), call `save_user_profile` to store them.\n"
    "- For eligibility questions ('am I eligible for X'): FIRST `retrieve` that visa's "
    "requirements, THEN compare them against the user's KNOWN PROFILE below.\n"
    "- If a required detail is missing from the profile, ASK the user for that specific "
    "detail instead of guessing. When they answer, save it and continue.\n"
    "- Give a clear verdict: which requirements are met, which are not, and the bottom line."
)


@dynamic_prompt
def system_prompt_with_profile(request) -> str:
    """Inject the live user profile into the system prompt on every turn."""
    profile = load_profile()
    profile_text = (
        json.dumps(profile, ensure_ascii=False, indent=2)
        if profile
        else "(no personal details saved yet)"
    )
    return f"{BASE_SYSTEM_PROMPT}\n\nKNOWN USER PROFILE:\n{profile_text}"


# ======================================================================
# The agent: tools + middleware (PII, summarization, profile) + memory
# ======================================================================
checkpointer = InMemorySaver()  # conversation memory (per thread)

agent = create_agent(
    llm,
    tools=[retrieve, save_user_profile],
    middleware=[
        # 1) PII: redact sensitive identifiers in user input so they never land
        #    in conversation history or logs. (Eligibility fields like nationality
        #    or deposit amount are NOT redacted - only emails / card numbers.)
        PIIMiddleware("email", strategy="redact"),
        PIIMiddleware("credit_card", strategy="redact"),
        # 2) Summarization: when the thread grows past ~16k tokens, summarize the
        #    older messages and keep the last 12 so we stay under qwen3's window.
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 16000),
            keep=("messages", 12),
        ),
        # 3) Profile injection: puts the saved personal details into the prompt.
        system_prompt_with_profile,
    ],
    checkpointer=checkpointer,
)

# Single local user -> one persistent conversation thread.
THREAD_ID = "local-user"
AGENT_CONFIG = {"configurable": {"thread_id": THREAD_ID}}


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
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q.question}]},
            config=AGENT_CONFIG,
        )
        logger.info("Finished processing chat request for question")
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
            # stream_mode="messages" yields (token, metadata) for every LLM call.
            # The agent's final answer comes from the "model" node. Tool-deciding
            # passes emit empty content (skipped by the token.content check), and
            # the summarization model runs in its own node (excluded by name), so
            # neither leaks into the streamed answer.
            for token, metadata in agent.stream(
                {"messages": [{"role": "user", "content": q.question}]},
                stream_mode="messages",
                config=AGENT_CONFIG,
            ):
                if metadata.get("langgraph_node") == "model" and token.content:
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
