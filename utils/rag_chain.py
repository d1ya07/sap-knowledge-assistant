"""
utils/rag_chain.py
100% LOCAL RAG pipeline using Ollama.
No data ever leaves your machine. No API keys needed.
Model: llama3.1 (runs on your Mac via Ollama)
"""

import logging
import os
import re
from typing import Dict, List, Any, Optional, Generator

import ollama
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "llama3.1"

SUPPORTED_LANGUAGES = {
    "English":    "English",
    "Hindi":      "Hindi",
    "Arabic":     "Arabic",
    "French":     "French",
    "German":     "German",
    "Spanish":    "Spanish",
    "Chinese":    "Chinese (Simplified)",
    "Japanese":   "Japanese",
    "Portuguese": "Portuguese",
    "Urdu":       "Urdu",
}

SYSTEM_PROMPT = """You are an expert SAP Support Assistant helping employees with SAP ERP questions.

Your STRICT rules:
1. Answer ONLY using the context provided below from official SAP documentation.
2. If the answer is not in the context, respond EXACTLY with:
   "This information is not available in the uploaded SAP documentation."
3. Never invent or guess any SAP transaction codes, menu paths, or process steps.
4. Be precise, structured, and professional.
5. When describing steps, use numbered lists.
6. Always mention transaction codes like ME21N or FB60 when they appear in context.
7. Use markdown formatting: ## for section headers, **bold** for key terms, `code` for T-codes.
8. At the very end of your answer, on a new line, write exactly:
   FOLLOWUPS: <question1> | <question2> | <question3>
   where each question is a short (under 10 words) natural follow-up.
   Do not add any other text after this line.
9. Answer in {language}."""


def configure_gemini() -> None:
    """Kept for compatibility — Ollama needs no API key."""
    try:
        ollama.list()
        logger.info("Ollama is running and accessible.")
    except Exception as e:
        raise EnvironmentError(
            "Ollama is not running. Please open the Ollama app on your Mac first."
        )


def _build_context(chunks: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Excerpt {i} | File: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def _extract_sources(chunks: List[Document]) -> List[Dict[str, Any]]:
    seen, sources = set(), []
    for doc in chunks:
        source = doc.metadata.get("source_file", "Unknown")
        page   = doc.metadata.get("page", "?")
        key    = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "page": page})
    return sources


def _build_messages(context: str, question: str, language: str,
                    chat_history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build Ollama-format messages list."""
    system_content = SYSTEM_PROMPT.format(language=language)
    system_content += f"\n\nContext from SAP Documentation:\n{'─'*40}\n{context}\n{'─'*40}"

    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        for turn in chat_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def _call_groq(messages: List[Dict], model_name: Optional[str] = None) -> str:
    """Non-streaming call via Ollama (kept name for compatibility)."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
        )
        return response['message']['content'].strip()
    except Exception as exc:
        logger.error(f"Ollama error: {exc}")
        return (
            "⚠️ Could not get a response from Ollama. "
            "Make sure the Ollama app is running on your Mac.\n\n"
            f"(Details: {exc})"
        )


def stream_gemini(messages, model_name: Optional[str] = None) -> Generator[str, None, None]:
    """Streaming call via Ollama (kept name for compatibility)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.get('message', {}).get('content', '')
            if text:
                yield text
    except Exception as exc:
        logger.error(f"Ollama streaming error: {exc}")
        yield (
            "\n\n⚠️ Could not reach Ollama. "
            "Make sure the Ollama app is open on your Mac.\n\n"
            f"(Details: {exc})"
        )


def _split_followups(raw_answer: str):
    followups: List[str] = []
    answer = raw_answer
    match  = re.search(r'FOLLOWUPS:\s*(.+)', raw_answer, re.IGNORECASE)
    if match:
        line      = match.group(1).strip()
        parts     = re.split(r'\||\n', line)
        followups = [p.strip().strip('-').strip() for p in parts if p.strip()][:3]
        answer    = raw_answer[:match.start()].rstrip()
    return answer, followups


def format_answer(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def ask_question_stream(
    question: str,
    vectorstore: FAISS,
    k: int = 5,
    model_name: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    language: str = "English",
) -> Dict[str, Any]:
    """Returns a dict with a streaming generator plus sources."""
    chunks = vectorstore.similarity_search(question, k=k)

    if not chunks:
        def _empty():
            yield "This information is not available in the uploaded SAP documentation."
        return {"stream": _empty(), "sources": [], "chunks": [], "followups": []}

    context  = _build_context(chunks)
    lang     = SUPPORTED_LANGUAGES.get(language, "English")
    messages = _build_messages(context, question, lang, chat_history)

    return {
        "stream":  stream_gemini(messages),
        "sources": _extract_sources(chunks),
        "chunks":  chunks,
    }


def summarise_document(
    chunks: List[Document],
    model_name: Optional[str] = None,
) -> str:
    sample   = chunks[:10]
    combined = "\n\n".join(c.page_content for c in sample)
    messages = [
        {"role": "system", "content": "You are an SAP documentation analyst. Be concise."},
        {"role": "user", "content": (
            "Summarise the following SAP document in 5-8 bullet points "
            "focusing on key processes, transaction codes, and modules covered.\n\n"
            f"Document content:\n{combined}\n\nSummary:"
        )},
    ]
    return _call_groq(messages)


def compare_documents(
    question: str,
    vectorstore: FAISS,
    doc_names: List[str],
    k_per_doc: int = 5,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare information across multiple documents for a given question/topic.
    Searches each document separately, then asks the AI to compare findings.
    """
    if len(doc_names) < 2:
        return {
            "answer": "Please select at least 2 documents to compare.",
            "per_doc_chunks": {},
        }

    # Search within each document separately using metadata filter
    per_doc_chunks: Dict[str, List[Document]] = {}
    for doc_name in doc_names:
        all_chunks = vectorstore.similarity_search(question, k=k_per_doc * 4)
        doc_chunks = [
            c for c in all_chunks
            if c.metadata.get("source_file") == doc_name
        ][:k_per_doc]
        per_doc_chunks[doc_name] = doc_chunks

    # Build comparison context
    context_parts = []
    for doc_name, chunks in per_doc_chunks.items():
        if not chunks:
            context_parts.append(f"=== {doc_name} ===\nNo relevant content found.\n")
            continue
        doc_text = "\n".join(c.page_content for c in chunks)
        context_parts.append(f"=== {doc_name} ===\n{doc_text}\n")

    full_context = "\n".join(context_parts)

    system_prompt = """You are an expert document analyst comparing information across multiple SAP documents.

Your task:
1. Compare how each document addresses the topic/question.
2. Highlight similarities between documents.
3. Highlight differences, contradictions, or unique details in each document.
4. Use a clear structure with headers for each document, then a comparison summary.
5. If a document doesn't contain relevant information, state that clearly.
6. Use markdown formatting with ## headers and **bold** for key terms.
7. Always mention transaction codes when they appear.

Be precise, structured, and professional."""

    user_prompt = (
        f"Compare the following documents on this topic: \"{question}\"\n\n"
        f"{full_context}\n\n"
        f"Provide a structured comparison."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    answer = _call_groq(messages, model_name)

    return {
        "answer": answer,
        "per_doc_chunks": per_doc_chunks,
        "doc_names": doc_names,
    }


def compare_documents_stream(
    question: str,
    vectorstore: FAISS,
    doc_names: List[str],
    k_per_doc: int = 5,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Streaming version of compare_documents."""
    if len(doc_names) < 2:
        def _err():
            yield "Please select at least 2 documents to compare."
        return {"stream": _err(), "per_doc_chunks": {}, "doc_names": doc_names}

    per_doc_chunks: Dict[str, List[Document]] = {}
    for doc_name in doc_names:
        all_chunks = vectorstore.similarity_search(question, k=k_per_doc * 4)
        doc_chunks = [
            c for c in all_chunks
            if c.metadata.get("source_file") == doc_name
        ][:k_per_doc]
        per_doc_chunks[doc_name] = doc_chunks

    context_parts = []
    for doc_name, chunks in per_doc_chunks.items():
        if not chunks:
            context_parts.append(f"=== {doc_name} ===\nNo relevant content found.\n")
            continue
        doc_text = "\n".join(c.page_content for c in chunks)
        context_parts.append(f"=== {doc_name} ===\n{doc_text}\n")

    full_context = "\n".join(context_parts)

    system_prompt = """You are an expert document analyst comparing information across multiple SAP documents.

Your task:
1. Compare how each document addresses the topic/question.
2. Highlight similarities between documents.
3. Highlight differences, contradictions, or unique details in each document.
4. Use a clear structure with headers for each document, then a comparison summary.
5. If a document doesn't contain relevant information, state that clearly.
6. Use markdown formatting with ## headers and **bold** for key terms.
7. Always mention transaction codes when they appear.

Be precise, structured, and professional."""

    user_prompt = (
        f"Compare the following documents on this topic: \"{question}\"\n\n"
        f"{full_context}\n\n"
        f"Provide a structured comparison."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return {
        "stream":         stream_gemini(messages, model_name),
        "per_doc_chunks": per_doc_chunks,
        "doc_names":      doc_names,
    }