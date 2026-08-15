"""
utils/pdf_loader.py
Handles PDF loading, text cleaning, and splitting into chunks.
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse more than 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalise bullet symbols
    text = re.sub(r'[•◦▪▸►‣]', '-', text)
    return text.strip()


def load_pdf(file_path: str) -> Tuple[List[Document], int]:
    """
    Load a single PDF file and return cleaned documents + page count.
    Each document = one page of the PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Loading PDF: {path.name}")

    try:
        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()
    except Exception as exc:
        logger.error(f"Failed to load {path.name}: {exc}")
        raise

    if not raw_docs:
        raise ValueError(f"No text could be extracted from: {path.name}")

    cleaned_docs: List[Document] = []
    for doc in raw_docs:
        cleaned = clean_text(doc.page_content)
        if cleaned:
            doc.page_content = cleaned
            doc.metadata["source_file"] = path.name
            cleaned_docs.append(doc)

    logger.info(f"Loaded {len(cleaned_docs)} pages from {path.name}")
    return cleaned_docs, len(raw_docs)


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """Load multiple PDFs and combine into one document list."""
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            docs, _ = load_pdf(fp)
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning(f"Skipping {fp}: {exc}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    Tries to split on paragraphs first, then sentences, then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks