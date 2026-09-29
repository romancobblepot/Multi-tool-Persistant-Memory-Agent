import os
import tempfile
from typing import Any, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from settings import embedding_model

_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict[str, Any]] = {}


def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(str(thread_id)) if thread_id else None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        docs = PyMuPDFLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embedding_model)
        _THREAD_RETRIEVERS[str(thread_id)] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_METADATA[str(thread_id)] = {"filename": filename or os.path.basename(temp_path), "documents": len(docs), "chunks": len(chunks)}
        return _THREAD_METADATA[str(thread_id)]
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def clear_thread_document(thread_id: str) -> dict:
    thread_id = str(thread_id)
    retriever = _THREAD_RETRIEVERS.pop(thread_id, None)
    metadata = _THREAD_METADATA.pop(thread_id, None)
    if retriever is None:
        return {"deleted": False, "message": "No book is attached to this chat."}
    return {"deleted": True, "message": f"Removed {metadata.get('filename', 'document')} and its FAISS index."}

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
