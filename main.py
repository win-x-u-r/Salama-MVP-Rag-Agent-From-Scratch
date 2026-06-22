"""Agentic RAG: LangGraph + OpenRouter, exposed via FastAPI.

Pattern: https://docs.langchain.com/oss/python/langgraph/agentic-rag
Run: uvicorn main:app --reload
"""
import glob
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

# --- LLM via OpenRouter (OpenAI-compatible API) ---------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# --- Knowledge base: load docs/ -> chunks -> in-memory vector store -------
def _load_chunks() -> list[Document]:
    docs = [
        Document(page_content=open(p, encoding="utf-8").read(), metadata={"source": p})
        for p in glob.glob("docs/**/*", recursive=True)
        if p.endswith((".txt", ".md"))
    ]
    if not docs:  # fallback so the template runs out of the box
        docs = [Document(page_content="LangGraph is a library for building stateful, "
                                       "multi-step agents as graphs.", metadata={"source": "seed"})]
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)


retriever = InMemoryVectorStore.from_documents(
    _load_chunks(), FastEmbedEmbeddings()
).as_retriever(search_kwargs={"k": 4})


@tool
def retrieve(query: str) -> str:
    """Search the knowledge base and return relevant passages."""
    return "\n\n".join(d.page_content for d in retriever.invoke(query))


# --- Nodes ----------------------------------------------------------------
def generate_query_or_respond(state: MessagesState):
    """Decide: answer directly, or call the retrieve tool."""
    return {"messages": [llm.bind_tools([retrieve]).invoke(state["messages"])]}


class Grade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Are the docs relevant? yes/no")


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question, context = state["messages"][0].content, state["messages"][-1].content
    score = llm.with_structured_output(Grade).invoke(
        f"Question: {question}\n\nRetrieved docs:\n{context}\n\n"
        "Are the docs relevant to the question? Answer yes or no."
    ).binary_score
    return "generate_answer" if score == "yes" else "rewrite_question"


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    better = llm.invoke(f"Rewrite this question to improve retrieval:\n\n{question}").content
    return {"messages": [{"role": "user", "content": better}]}


def generate_answer(state: MessagesState):
    question, context = state["messages"][0].content, state["messages"][-1].content
    return {"messages": [llm.invoke(
        f"Answer the question using only the context. Be concise (max 3 sentences).\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )]}


# --- Graph ----------------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node(generate_query_or_respond)
builder.add_node("retrieve", ToolNode([retrieve]))
builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges(
    "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
)
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("rewrite_question", "generate_query_or_respond")
builder.add_edge("generate_answer", END)

graph = builder.compile()


# --- FastAPI --------------------------------------------------------------
app = FastAPI(title="Agentic RAG")


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(q: Query):
    state = graph.invoke({"messages": [{"role": "user", "content": q.question}]})
    return {"answer": state["messages"][-1].content}
