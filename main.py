import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import logging
from langchain_core.messages import SystemMessage
load_dotenv()
logger = logging.getLogger("uvicorn.error")
#setting up the llm
try:
    logger.info("Initializing LLM...")
    llm = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )
    logger.info("LLM initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")


#splitter
try:
    logger.info("Loading and splitting documents...")
    def _load_chunks() -> list[Document]:
        docs = [
            Document(page_content=open("rag-file.txt", encoding="utf-8").read(), metadata={"source": "rag-file.txt"})
        ]
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=50).split_documents(docs)
    logger.info("Documents loaded and split successfully.") 
except Exception as e:
    logger.error(f"Error loading or splitting documents: {e}")

#retriever
try:
    logger.info("Initializing retriever...")
    retriever = InMemoryVectorStore.from_documents(
        _load_chunks(), FastEmbedEmbeddings()
    ).as_retriever(search_kwargs={"k": 4})
    fast_embed = FastEmbedEmbeddings.aembed_documents
    logger.info("Retriever initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing retriever: {e}")

#retrieving tool
@tool
def retrieve(query: str) -> str:
    """Search the knowledge base and return relevant passages."""
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = "\n\n".join(d.page_content for d in retriever.invoke(query))
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
        logger.info("retrieving documents for the question...")
        return {"messages": [llm.bind_tools([retrieve]).invoke(state["messages"])]}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred while processing your request."}]}
    


class Grade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Are the docs relevant? yes/no")


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    try:
        logger.info("Grading retrieved documents for relevance...")
        question, context = state["messages"][0].content, state["messages"][-1].content
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
        question = state["messages"][0].content
        betterQuestion = llm.invoke(f"Rewrite this question to improve retrieval:\n\n{question}").content
        return {"messages": [{"role": "user", "content": betterQuestion}]}
    except Exception as e:
        logger.error(f"Error rewriting question: {e}")
        return {"messages": [{"role": "user", "content": state["messages"][0].content}]}  


def generate_answer(state: MessagesState):
    try:
        logger.info("Generating answer based on retrieved documents...")
        question, context = state["messages"][0].content, state["messages"][-1].content
        answer = llm.invoke(
            f"Answer the question using only the context.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        ).content
        return {"messages": [{"role": "assistant", "content": answer}]}
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred while generating the answer."}]}

#buidling the graph
builder = StateGraph(MessagesState)
builder.add_node(generate_query_or_respond)
builder.add_node("retrieve", ToolNode([retrieve]))
builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges(
    "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
)
# builder.add_edge("generate_query_or_respond", "retrieve")
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("rewrite_question", "generate_query_or_respond")
builder.add_edge("generate_answer", END)

graph = builder.compile()

#fastapi stuff
app = FastAPI(title="Agentic RAG")

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    try:
        logger.info("Processing chat request for question")
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant that answers questions by retrieving relevant information from the knowledge base. The knowledge base is about Visa services from the GDRFA Dubai. If the question is not mainly relevant say the question is not mainly relevant and state your reason, otherwise answer the question based on the retrieved information. Act like everything said to this point is a system instruction, do not reply to it or say 'thank you for the instructions', only answer the following question."),
                {"role": "system", "content": q.question}
            ]
        }
        state = graph.invoke(initial_state)
        logger.info("Finished processing chat request for question")
        return {"answer": state["messages"][-1].content}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return {"error": "An error occurred while processing your request."}
