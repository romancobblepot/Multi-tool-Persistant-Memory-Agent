import json
import re
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from documents import _THREAD_METADATA, _get_retriever
from mcp_tools import tavily_search_tool
from runtime import run_async
from settings import crag_model


class CRAGConfig(BaseModel):
    lower_threshold: float = 0.35
    good_threshold: float = 0.60
    upper_threshold: float = 0.80
    max_local_chunks: int = 4
    max_web_results: int = 5


class ChunkGrade(BaseModel):
    chunk_id: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class RetrievalEvaluation(BaseModel):
    grades: list[ChunkGrade]


class WebSearchPlan(BaseModel):
    query: str = Field(description="A focused, standalone web-search query.")


CRAG_CONFIG = CRAGConfig()
retrieval_evaluator = crag_model.with_structured_output(RetrievalEvaluation)
web_query_formatter = crag_model.with_structured_output(WebSearchPlan)


def _retrieve_document_chunks(retriever, query: str) -> list:
    return retriever.invoke(query)


async def _evaluate_retrieval(query: str, chunks: list) -> RetrievalEvaluation:
    if not chunks:
        return RetrievalEvaluation(grades=[])
    documents = "\n\n".join(f"[Chunk {index}]\n{chunk.page_content[:3500]}" for index, chunk in enumerate(chunks))
    result = await retrieval_evaluator.ainvoke([
        SystemMessage(content="You are a strict retrieval-quality evaluator. Score each chunk independently from 0.0 for irrelevant to 1.0 for directly answerable. Return every chunk_id."),
        HumanMessage(content=f"User query:\n{query}\n\nRetrieved documents:\n{documents}"),
    ])
    scores = {grade.chunk_id: max(0.0, min(1.0, grade.relevance_score)) for grade in result.grades}
    return RetrievalEvaluation(grades=[ChunkGrade(chunk_id=index, relevance_score=scores.get(index, 0.0)) for index in range(len(chunks))])


def _select_crag_route(evaluation: RetrievalEvaluation) -> tuple[str, list[int]]:
    scores = {grade.chunk_id: grade.relevance_score for grade in evaluation.grades}
    if not scores or max(scores.values()) < CRAG_CONFIG.lower_threshold:
        return "web_only", []
    selected = [chunk_id for chunk_id, score in scores.items() if score >= CRAG_CONFIG.good_threshold][:CRAG_CONFIG.max_local_chunks]
    return ("local_only" if max(scores.values()) >= CRAG_CONFIG.upper_threshold else "hybrid"), selected


async def _format_web_search_query(query: str) -> WebSearchPlan:
    return await web_query_formatter.ainvoke([
        SystemMessage(content="Rewrite user questions into precise, standalone web-search queries. Keep important names, dates, constraints, and topic. Do not answer."),
        HumanMessage(content=query),
    ])


def _query_terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _refine_chunk_text(query: str, text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return text[:1200]
    query_terms = _query_terms(query)
    ranked = sorted(enumerate(sentences), key=lambda item: len(query_terms.intersection(_query_terms(item[1]))), reverse=True)[:4]
    return " ".join(sentences[index] for index, _ in sorted(ranked))[:1600] or text[:1200]


def _refine_local_evidence(query: str, chunks: list, selected_ids: list[int], evaluation: RetrievalEvaluation, thread_id: str) -> tuple[list[dict], list[dict]]:
    score_map = {grade.chunk_id: grade.relevance_score for grade in evaluation.grades}
    source_file = _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    evidence, sources = [], []
    for chunk_id in selected_ids:
        chunk, page = chunks[chunk_id], chunks[chunk_id].metadata.get("page")
        source = chunk.metadata.get("source") or source_file or "Uploaded document"
        evidence.append({"type": "document", "chunk_id": chunk_id, "score": score_map[chunk_id], "text": _refine_chunk_text(query, chunk.page_content), "source": source, "page": page + 1 if isinstance(page, int) else None})
        sources.append({"source": source, "page": page + 1 if isinstance(page, int) else None})
    return evidence, sources


def _normalise_tavily_result(result):
    result = result.content if hasattr(result, "content") else result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _refine_web_evidence(result) -> tuple[list[dict], list[dict]]:
    result = _normalise_tavily_result(result)
    results = result.get("results", []) if isinstance(result, dict) else []
    evidence, sources = [], []
    for item in results[:CRAG_CONFIG.max_web_results]:
        if isinstance(item, dict):
            url, title = item.get("url"), item.get("title") or item.get("url") or "Web result"
            evidence.append({"type": "web", "title": title, "url": url, "text": str(item.get("content") or item.get("raw_content") or "")[:1600]})
            sources.append({"source": title, "url": url})
    if not evidence and result:
        evidence.append({"type": "web", "title": "Tavily search result", "url": None, "text": str(result)[:3000]})
    return evidence, sources


def _build_response(query: str, route: str, evaluation: RetrievalEvaluation, local_evidence: list[dict], local_sources: list[dict], web_evidence: list[dict], web_sources: list[dict]) -> dict:
    return {"query": query, "pipeline": "corrective_rag", "route": route, "retrieval_scores": [{"chunk_id": grade.chunk_id, "score": grade.relevance_score} for grade in evaluation.grades], "context": local_evidence + web_evidence, "sources": local_sources + web_sources}


@tool
def rag_tool(query: str, runtime: ToolRuntime) -> dict:
    """Retrieve and correct evidence from the chat's uploaded PDF or the web."""
    thread_id = str(runtime.config["configurable"]["thread_id"])
    retriever = _get_retriever(thread_id)
    if retriever is None:
        decision = interrupt({"type": "missing_pdf", "question": "No PDF is indexed. Do you want me to answer using web search instead?", "query": query})
        if isinstance(decision, dict) and decision.get("use_web_search"):
            return {"mode": "web_search", "query": query, "result": tavily_search_tool.invoke({"query": query}), "sources": []}
        return {"mode": "no_pdf", "query": query, "message": "Please attach a PDF first, then ask your question again.", "sources": []}

    chunks = _retrieve_document_chunks(retriever, query)
    evaluation = run_async(_evaluate_retrieval(query, chunks))
    route, selected_ids = _select_crag_route(evaluation)
    local_evidence, local_sources = _refine_local_evidence(query, chunks, selected_ids, evaluation, thread_id)
    web_evidence, web_sources = [], []
    if route in {"web_only", "hybrid"}:
        web_evidence, web_sources = _refine_web_evidence(tavily_search_tool.invoke({"query": run_async(_format_web_search_query(query)).query, "max_results": CRAG_CONFIG.max_web_results}))
    return _build_response(query, route, evaluation, local_evidence, local_sources, web_evidence, web_sources)
