import os
import logging
import time

from dotenv import load_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from .splitter_setup import _load_chunks

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
        disable_streaming="tool_calling" # this apparently fixes the tool call being called with every chunk 
    )
    logger.info("LLM initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")

# The clarifier is a simple 3-way judgment; a small model does it in a fraction
# of the time. Set CLARIFIER_MODEL (e.g. qwen/qwen3-8b) to enable; defaults to
# the main model.
clarifier_llm = llm
_clarifier_model = os.getenv("CLARIFIER_MODEL")
if _clarifier_model:
    try:
        clarifier_llm = ChatOpenRouter(
            model=_clarifier_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
            reasoning={"effort": "none"},
            openrouter_provider={"sort": "throughput"},
            seed=42,
            disable_streaming="tool_calling",
        )
        logger.info(f"Clarifier LLM: {_clarifier_model}")
    except Exception as e:
        logger.error(f"Error initializing clarifier LLM, falling back to main: {e}")
        clarifier_llm = llm

# old retriever that used to be with fast embed and in memory vector store
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
    # tiktoken doesn't know the OpenRouter-prefixed model name; without this it
    # warns "model not found. Using cl100k_base" on every embed call
    tiktoken_model_name="text-embedding-3-large",
)# some workaround to route the openai embedding model through openrouter

# ---- long-term memory store + durable conversation checkpointer (Postgres) ----
# PostgresStore: semantic long-term memory (facts, per-user namespaces).
# PostgresSaver: conversations survive restarts (replaces InMemorySaver).
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = os.getenv("DB_URI", "")
store = None
try:
    logger.info("Setting up Postgres memory store...")
    _t0 = time.perf_counter()
    # sync pool: used by the store from (sync) graph nodes, and for one-time DDL
    _pg_pool = ConnectionPool(
        DB_URI,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    _pg_pool.wait(timeout=10)  # fail fast if Postgres is unreachable
    store = PostgresStore(
        _pg_pool,
        index={
            "embed": embeddings,
            "dims": 3072,
            # flat = exact scan: pgvector's HNSW caps at 2000 dims, and a
            # per-user facts namespace is far too small to need ANN anyway
            "ann_index_config": {"kind": "flat"},
        },
    )
    store.setup()
    PostgresSaver(_pg_pool).setup()  # create checkpoint tables (shared schema)
    # NOTE: the durable checkpointer itself is AsyncPostgresSaver, attached in
    # main.py's startup hook — its pool cannot be built without a running
    # event loop. Until startup completes, the graph holds InMemorySaver.
    logger.info(
        f"Postgres store ready. [timing] setup: {time.perf_counter() - _t0:.2f}s"
    )
except Exception as e:
    logger.error(f"Error setting up Postgres store (long-term memory disabled): {e}")
    store = None

RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "qdrant").lower()

graph_retriever = None
if RETRIEVER_BACKEND == "graph":
    try:
        logger.info("setting up Neo4j graph retriever")
        _t0 = time.perf_counter()
        from graph_retriever import GraphRetriever

        graph_retriever = GraphRetriever(embeddings)
        logger.info(f"Neo4j graph retriever ready. [timing] setup: {time.perf_counter() - _t0:.2f}s")
    except Exception as e:
        logger.error(f"Error setting up Neo4j graph retriever: {e}")

#Qdrant setup
qdrant = None
if RETRIEVER_BACKEND == "qdrant":
    try:
        # use existing collection:
        logger.info("setting up Qdrant client")
        _t0 = time.perf_counter()
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="rag_collection",
            path=("./qdrant_collection"),
        )
        logger.info(f"qdrant succesfully setup [timing] setup: {time.perf_counter() - _t0:.2f}s")
    except Exception as e:
        logger.error(f"error setting up qdrant: {e}")

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