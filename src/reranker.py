import logging
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents.compressor import BaseDocumentCompressor

logger = logging.getLogger("uvicorn.error")

RERANK_MODEL = "jinaai/jina-reranker-v1-turbo-en"  # 8K context

try:
    logger.info(f"Loading reranker: {RERANK_MODEL}")
    _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    logger.info("Reranker ready.")
except Exception as e:
    logger.error(f"Error loading reranker: {e}")
    _reranker = None


class FastEmbedRerank(BaseDocumentCompressor):
    """Cross-encoder reranker compressor (FastEmbed, ONNX/CPU)."""

    top_n: int = 3

    def compress_documents(self, documents, query, callbacks=None):
        docs = list(documents)
        if not docs or _reranker is None:
            return docs[: self.top_n]                       # fall back to recall order
        scores = list(_reranker.rerank(query, [d.page_content for d in docs]))  # score per doc
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[: self.top_n]]

 