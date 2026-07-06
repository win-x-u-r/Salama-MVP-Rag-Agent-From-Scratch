import json
import logging

from langchain_core.documents import Document

logger = logging.getLogger("uvicorn.error")

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
