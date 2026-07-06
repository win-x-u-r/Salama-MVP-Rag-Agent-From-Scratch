import os
import logging
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.agent_core import GRAPH_CONFIG, PROFILE_PATH, graph, load_profile

logger = logging.getLogger("uvicorn.error")

#fastapi stuff
app = FastAPI(title="Agentic RAG")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html"))


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
                if metadata.get("langgraph_node") == "responder agent" and token.content:
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
