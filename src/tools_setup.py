import os
import json
import logging
import time

from langchain_core.tools import tool

from .backend_setup import RETRIEVER_BACKEND, graph_retriever, embeddings, qdrant

logger = logging.getLogger("uvicorn.error")

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
