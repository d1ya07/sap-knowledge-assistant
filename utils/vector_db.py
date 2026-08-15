"""
utils/vector_db.py
Manages the FAISS vector store: build, save, load, search, merge.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = Path("vectorstore")


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model (downloads once, ~90MB)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Take document chunks, embed them, and create a FAISS index.
    This is what lets us do semantic search later.
    """
    if not chunks:
        raise ValueError("No chunks provided to build_vectorstore.")
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully.")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Optional[str] = None) -> str:
    """Save FAISS index to disk so it persists between app restarts."""
    save_dir = Path(path) if path else VECTORSTORE_PATH
    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    logger.info(f"Vectorstore saved to: {save_dir}")
    return str(save_dir)


def load_vectorstore(path: Optional[str] = None) -> Optional[FAISS]:
    """Load a previously saved FAISS index from disk."""
    load_dir = Path(path) if path else VECTORSTORE_PATH
    index_file = load_dir / "index.faiss"

    if not index_file.exists():
        logger.info("No existing vectorstore found.")
        return None

    logger.info(f"Loading vectorstore from: {load_dir}")
    embeddings = _get_embeddings()
    try:
        vectorstore = FAISS.load_local(
            str(load_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception as exc:
        logger.error(f"Failed to load vectorstore: {exc}")
        return None


def vectorstore_exists(path: Optional[str] = None) -> bool:
    """Check if a saved FAISS index exists on disk."""
    load_dir = Path(path) if path else VECTORSTORE_PATH
    return (load_dir / "index.faiss").exists()


def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """Find the top-k most relevant document chunks for a query."""
    return vectorstore.similarity_search(query, k=k)


def merge_vectorstores(existing: FAISS, new_store: FAISS) -> FAISS:
    """Merge a new index into the existing one (for adding more PDFs)."""
    existing.merge_from(new_store)
    logger.info("Vectorstores merged.")
    return existing