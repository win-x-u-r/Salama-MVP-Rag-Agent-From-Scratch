import logging
import time

from langchain_core.tools import tool
from langchain_classic.retrievers import ContextualCompressionRetriever
from .reranker import FastEmbedRerank
from .backend_setup import RETRIEVER_BACKEND, graph_retriever, embeddings, qdrant
from .config import RETRIEVAL_K, RERANKER_TOP_N

logger = logging.getLogger("uvicorn.error")

#retrieving tool
# content_and_artifact: the passages remain the tool result the LLM sees; the
# artifact carries the matched services' names/scores for the panel node.
@tool(response_format="content_and_artifact")
def retrieve(query: str):
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
        t0 = time.perf_counter()
        compressed_docs = FastEmbedRerank(top_n=RERANKER_TOP_N).compress_documents(retrieved_docs, query)
        logger.info(f"Compressed to {len(compressed_docs)} documents for query: {query},\n [timing] Reranking: {time.perf_counter()-t0:.2f}s")
        logger.info(f"Documents retrieved and compressed successfully with k={k} and top_n={topN}")
        artifact = {
            "services": [
                {"name": d.metadata.get("name"), "score": d.metadata.get("score")}
                for d in compressed_docs
                if d.metadata.get("name")
            ]
        }
        return [d.page_content for d in compressed_docs], artifact
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return "An error occurred while retrieving documents.", {}


# Long-term memory now lives in the Postgres store (see backend_setup.py):
# written by the graph's `memory writer` node, recalled semantically by the
# responder, managed via the /profile endpoints.
TOOLS = [retrieve]
