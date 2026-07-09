import os
import json
import logging
import time

from langchain_core.tools import tool
from langchain_classic.retrievers import ContextualCompressionRetriever
from .reranker import FastEmbedRerank
from .backend_setup import RETRIEVER_BACKEND, graph_retriever, embeddings, qdrant
from .config import RETRIEVAL_K, RERANKER_TOP_N

logger = logging.getLogger("uvicorn.error")

#retrieving tool
@tool
def retrieve(query: str) -> str:
    """Search the knowledge base and return relevant passages.

    Args:
        query (str): The user's question or query.
    """
    try:
        logger.info(
            f"Retrieving documents for query: {query}"
            f"| backend={RETRIEVER_BACKEND}"
        )
        k = RETRIEVAL_K
        topN = RERANKER_TOP_N
        if RETRIEVER_BACKEND == "graph" and graph_retriever is not None:
            graph_retriever.k = k
            base = graph_retriever
        else:
            base = qdrant.as_retriever(search_kwargs={"k": k})
            # query_vector = embeddings.embed_query(query)
            # docs = qdrant.similarity_search_by_vector(query_vector, k=k)
            # results = "\n\n".join(d.page_content for d in docs)

        # retriever = ContextualCompressionRetriever(
        #     base_compressor=FastEmbedRerank(top_n=topN),
        #     base_retriever=base,
        # )

        t0 = time.perf_counter()
        retrieved_docs = base.invoke(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query},\n [timing] Retrieval: {time.perf_counter()-t0:.2f}s")
        compressed_docs = FastEmbedRerank(top_n=RERANKER_TOP_N).compress_documents(retrieved_docs, query)
        logger.info(f"Compressed to {len(compressed_docs)} documents for query: {query},\n [timing] Reranking: {time.perf_counter()-t0:.2f}s")
        logger.info(f"[timing] retrieve and compress total={time.perf_counter() - t0:.2f}s")
        logger.info(f"Documents retrieved and compressed successfully with k={k} and top_n={topN}")
        return [d.page_content for d in compressed_docs]
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return "An error occurred while retrieving documents."


# long term memory (for personal context)
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_profile.json")


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


TOOLS = [retrieve, save_user_profile, clear_user_profile]
