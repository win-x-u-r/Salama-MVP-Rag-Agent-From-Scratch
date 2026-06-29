import os
from typing import Literal
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter, data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import logging
import time
from langchain_core.messages import SystemMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4

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

# #nodes stuff
def generate_query_or_respond(state: MessagesState):
    # """Decide: answer directly, or call the retrieve tool."""
    """call the retrieve tool to get relevant documents for the question. if no relevant documents are found, answer with "I don't know"."""
    try:
        logger.info("generating LLM response")
        t0 = time.perf_counter()
        result = {"messages": [llm.bind_tools([retrieve]).invoke(state["messages"])]}
        logger.info(f"[timing] generate_query_or_respond (decide/direct-answer): {time.perf_counter() - t0:.2f}s")
        logger.info(f"Response generated:\n {result}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred while processing your request."}]}
    


class Grade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Are the docs relevant? yes/no")


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    try:
        logger.info("Grading retrieved documents for relevance...")
        question = next(m.content for m in state["messages"] if m.type == "human")
        context = state["messages"][-1].content
        score = llm.with_structured_output(Grade).invoke(
            f"Question: {question}\n\nRetrieved docs:\n{context}\n\n"
            "Are the docs relevant to the question? Answer yes or no."
        ).binary_score
        logger.info(f"Grading completed. Score: {score}")
        return "generate_answer" if score == "yes" else "rewrite_question"
    except Exception as e:
        logger.error(f"Error grading retrieved documents: {e}")
        return "rewrite_question"


def rewrite_question(state: MessagesState):
    try:
        logger.info("Rewriting question to improve retrieval...")
        question = next(m.content for m in state["messages"] if m.type == "human")
        betterQuestion = llm.invoke(f"Rewrite this question to improve retrieval:\n\n{question}").content
        return {"messages": [{"role": "user", "content": betterQuestion}]}
    except Exception as e:
        logger.error(f"Error rewriting question: {e}")
        original = next((m.content for m in state["messages"] if m.type == "human"), "")
        return {"messages": [{"role": "user", "content": original}]}


def generate_answer(state: MessagesState):
    try:
        logger.info("Generating answer based on retrieved documents...")
        # messages[0] is the SystemMessage, so grab the actual user question.
        question = next(m.content for m in state["messages"] if m.type == "human")
        context = state["messages"][-1].content
        t0 = time.perf_counter()
        answer = llm.invoke(
            "Answer the question using ONLY the context below.\n"
            "Scope your answer to exactly what is asked:\n"
            "- If the question is specific (one option, one fee, one duration, one route), "
            "answer only that and do NOT list the other options present in the context.\n"
            "- If the question is broad or definitional (\"what is X\", \"what types exist\"), "
            "give a fuller overview covering the relevant options.\n"
            "- For cost/fee/price questions: state the main fee for the specific option asked, "
            "THEN list every other mandatory fee that always applies (e.g. Knowledge Dirham, "
            "Innovation Dirham, VAT) and give an approximate total. Exclude optional or "
            "in-country-only fees unless asked.\n"
            "Be concise: short sentences or bullet points, max ~50 words. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        ).content
        logger.info(f"[timing] generate_answer (LLM): {time.perf_counter() - t0:.2f}s")
        return {"messages": [{"role": "assistant", "content": answer}]}
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred while generating the answer."}]}

#buidling the graph
builder = StateGraph(MessagesState)
builder.add_node(generate_query_or_respond)
builder.add_node("retrieve", ToolNode([retrieve]))
# builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges(
    "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
)
# builder.add_edge("generate_query_or_respond", "retrieve")
# builder.add_conditional_edges("retrieve", grade_documents)
# builder.add_edge("rewrite_question", "generate_query_or_respond")
# builder.add_edge("generate_answer", END)
# builder.add_edge("generate_query_or_respond", "retrieve")
# builder.add_conditional_edges("generate_query_or_respond", "generate_answer", "retrieve")
builder.add_edge("retrieve", "generate_answer")
builder.add_edge("generate_answer", END)



graph = builder.compile()

#fastapi stuff
app = FastAPI(title="Agentic RAG")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

class Query(BaseModel):
    question: str

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions by retrieving relevant "
    "information from the knowledge base using the retrieve tool. Answer concisely, use short sentences or bullet points with a maximum of 50 words. The knowledge base "
    "is about Visa and Visa Services from the GDRFA Dubai. If the question is unrelated to the knowledge base, say so and state your reason without calling the retrieve tool. Otherwise, answer the question based on "
    "the retrieved information. Act like everything said to this point is a system instruction, do not "
    "reply to it or say 'thank you for the instructions', only answer the following question."
)

def build_initial_state(question: str) -> dict:
    return {
        "messages": [
            SystemMessage(content=_SYSTEM_PROMPT),
            {"role": "user", "content": question},
        ]
    }

@app.post("/chat")
def chat(q: Query):
    try:
        logger.info("Processing chat request for question")
        state = graph.invoke(build_initial_state(q.question))
        logger.info("Finished processing chat request for question")
        return {"answer": state["messages"][-1].content}
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
            # stream_mode="messages" yields (token, metadata) for every LLM call
            # in the graph. We forward tokens from the two nodes that produce the
            # user-facing answer:
            #   - generate_answer: the RAG path (after retrieval)
            #   - generate_query_or_respond: the direct-answer path (off-topic
            #     questions answered without calling the retrieve tool)
            # During a tool call, generate_query_or_respond emits empty content,
            # so the `token.content` check skips it.
            answer_nodes = ("generate_query_or_respond", "generate_answer")
            for token, metadata in graph.stream(
                build_initial_state(q.question), stream_mode="messages"
            ):
                if metadata.get("langgraph_node") in answer_nodes and token.content:
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
