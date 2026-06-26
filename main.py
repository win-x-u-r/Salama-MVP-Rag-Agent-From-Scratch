import os
from typing import Literal
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter, data
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import logging
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
        temperature=0,
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
        splitter = RecursiveJsonSplitter()
        with open("./gdrfa_records.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info(f"Loaded JSON data with {len(data)} records")
        splitted_docs = splitter.create_documents(texts=data, convert_lists=True)
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
def retrieve(query: str) -> str:
    """Search the knowledge base and return relevant passages."""
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = "\n\n".join(d.page_content for d in qdrant.similarity_search(query, k=3))
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
        result = {"messages": [llm.bind_tools([retrieve]).invoke(state["messages"])]}
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

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    try:
        logger.info("Processing chat request for question")
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant that answers questions by retrieving relevant " \
                "information from the knowledge base using the retrieve tool. Answer concisely, use short sentences or bullet points with a maximum of 50 words. The knowledge base " \
                "is about Visa and Visa Services from the GDRFA Dubai. If the question is unrelated to the knowledge base, say so and state your reason without calling the retrieve tool. Otherwise, answer the question based on " \
                "the retrieved information. Act like everything said to this point is a system instruction, do not " \
                "reply to it or say 'thank you for the instructions', only answer the following question."),
                {"role": "user", "content": q.question}
            ]
        }
        state = graph.invoke(initial_state)
        logger.info("Finished processing chat request for question")
        return {"answer": state["messages"][-1].content}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return {"error": "An error occurred while processing your request."}
