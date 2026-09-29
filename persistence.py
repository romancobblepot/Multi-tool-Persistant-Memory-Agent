import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel, Field
from runtime import run_async
from settings import SUPABASE_DB_URL, embedding_model, model
_checkpointer_context = None
_store_context = None


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(embedding_model.embed_documents, texts)


async def _init_persistence():
    global _checkpointer_context, _store_context
    _checkpointer_context = AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL)
    checkpointer = await _checkpointer_context.__aenter__()
    await checkpointer.setup()
    _store_context = AsyncPostgresStore.from_conn_string(SUPABASE_DB_URL, index={
        "dims": 768, "embed": _embed_texts, "fields": ["text"], "distance_type": "cosine",
        "ann_index_config": {"kind": "hnsw", "m": 16, "ef_construction": 64},
    })
    memory_store = await _store_context.__aenter__()
    await memory_store.setup()
    return checkpointer, memory_store


checkpointer, memory_store = run_async(_init_persistence())
_PROFILE_NAMESPACE = ("user_profiles",)
THREAD_METADATA_NAMESPACE = "thread_metadata"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _profile_name(display_name: str) -> str:
    return (display_name or "").strip()[:80] or "Unnamed user"


async def _create_user_profile(display_name: str) -> dict:
    user_id, now = str(uuid.uuid4()), _now()
    profile = {"display_name": _profile_name(display_name), "created_at": now, "updated_at": now}
    await memory_store.aput(_PROFILE_NAMESPACE, user_id, profile, index=False)
    return {"user_id": user_id, **profile}


def create_user_profile(display_name: str) -> dict:
    return run_async(_create_user_profile(display_name))


async def _ensure_user_profile(user_id: str, display_name: str = "User") -> dict:
    existing = await memory_store.aget(_PROFILE_NAMESPACE, str(user_id))
    if existing is not None:
        return {"user_id": existing.key, **existing.value}
    now = _now()
    profile = {"display_name": _profile_name(display_name), "created_at": now, "updated_at": now}
    await memory_store.aput(_PROFILE_NAMESPACE, str(user_id), profile, index=False)
    return {"user_id": str(user_id), **profile}


def ensure_user_profile(user_id: str, display_name: str = "User") -> dict:
    return run_async(_ensure_user_profile(user_id, display_name))


async def _list_user_profiles() -> list[dict]:
    items = await memory_store.asearch(_PROFILE_NAMESPACE, limit=100)
    return [{"user_id": item.key, **item.value} for item in items if isinstance(item.value, dict)]


def list_user_profiles() -> list[dict]:
    return run_async(_list_user_profiles())


def _thread_namespace(user_id: str) -> tuple[str, str]:
    return (THREAD_METADATA_NAMESPACE, str(user_id))


def _clean_thread_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "")).strip().strip("\"'`:- ")[:60] or "New chat"


async def _create_thread_metadata(user_id: str, thread_id: str, title: str = "New chat", title_source: str = "default") -> dict:
    namespace = _thread_namespace(user_id)
    existing = await memory_store.aget(namespace, str(thread_id))
    if existing is not None:
        return {"thread_id": existing.key, **existing.value}
    now = _now()
    value = {"title": _clean_thread_title(title), "title_source": title_source, "created_at": now, "updated_at": now}
    await memory_store.aput(namespace, str(thread_id), value, index=False)
    return {"thread_id": str(thread_id), **value}


def create_thread_metadata(user_id: str, thread_id: str, title: str = "New chat") -> dict:
    return run_async(_create_thread_metadata(user_id, thread_id, title))


async def _get_thread_metadata(user_id: str, thread_id: str) -> dict:
    item = await memory_store.aget(_thread_namespace(user_id), str(thread_id))
    return {"thread_id": str(thread_id), "title": "New chat", "title_source": "default"} if item is None else {"thread_id": item.key, **item.value}


def get_thread_metadata(user_id: str, thread_id: str) -> dict:
    return run_async(_get_thread_metadata(user_id, thread_id))


async def _list_thread_metadata(user_id: str) -> dict[str, dict]:
    items = await memory_store.asearch(_thread_namespace(user_id), limit=100)
    return {str(item.key): {"thread_id": item.key, **item.value} for item in items}


def list_thread_metadata(user_id: str) -> dict[str, dict]:
    return run_async(_list_thread_metadata(user_id))


async def _update_thread_title(user_id: str, thread_id: str, title: str, title_source: str) -> dict:
    current = await _get_thread_metadata(user_id, thread_id)
    value = {**{key: value for key, value in current.items() if key != "thread_id"}, "title": _clean_thread_title(title), "title_source": title_source, "updated_at": _now()}
    await memory_store.aput(_thread_namespace(user_id), str(thread_id), value, index=False)
    return {"thread_id": str(thread_id), **value}


def rename_thread(user_id: str, thread_id: str, title: str) -> dict:
    return run_async(_update_thread_title(user_id, thread_id, title, "manual"))


async def generate_and_save_thread_title(user_id: str, thread_id: str, first_prompt: str) -> None:
    if (await _get_thread_metadata(user_id, thread_id)).get("title_source") in {"auto", "manual"}:
        return
    try:
        response = await model.ainvoke("Generate a 3–6 word chat title. Return only the title. Do not use Chat or Conversation.\n\n" f"User message:\n{first_prompt[:1500]}")
        if _clean_thread_title(str(response.content)).lower() != "new chat":
            if (await _get_thread_metadata(user_id, thread_id)).get("title_source") != "manual":
                await _update_thread_title(user_id, thread_id, str(response.content), "auto")
    except Exception as error:
        print(f"Thread title generation failed: {error}")


def _user_id(config: RunnableConfig) -> str:
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        raise ValueError("configurable.user_id is required for long-term memory.")
    return str(user_id)


def latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content.strip() if isinstance(message.content, str) else str(message.content).strip()
    return ""


async def retrieve_long_term_memory(state: dict, config: RunnableConfig):
    query = latest_human_text(state.get("messages", []))
    if not query:
        return {"user_memories": []}
    results = await memory_store.asearch(("user_memories", _user_id(config)), query=query, limit=2)
    return {"user_memories": [item.value["text"] for item in results if isinstance(item.value, dict) and item.value.get("text")]}


_IGNORED_MEMORY_TEXT = {"yes", "no", "y", "n", "okay", "ok", "thanks", "thank you"}


class MemoryExtraction(BaseModel):
    memories: list[str] = Field(default_factory=list)


class MemoryDecision(BaseModel):
    action: Literal["create", "update", "skip"]
    memory_key: Optional[str] = None
    text: Optional[str] = None


async def _extract_durable_memories(user_message: str) -> list[str]:
    extractor = model.with_structured_output(MemoryExtraction)
    result = await extractor.ainvoke(f"""Extract at most two durable user memories from this message. Keep only stable preferences, background, goals, constraints, or ongoing projects. Exclude ordinary questions, one-off requests, greetings, and yes/no replies. Write concise third-person facts.\n\nUser message:\n{user_message}""")
    return list(dict.fromkeys(item.strip() for item in result.memories if item.strip()))[:2]


async def _decide_memory_action(candidate: str, matches: list[Any]) -> MemoryDecision:
    match_text = "\n".join(f"- key={item.key!r}; text={item.value.get('text', '')!r}; score={item.score}" for item in matches if isinstance(item.value, dict)) or "- No similar memories."
    manager = model.with_structured_output(MemoryDecision)
    decision = await manager.ainvoke(f"""You manage long-term user memory. Candidate: {candidate}\n\nMatches:\n{match_text}\n\nChoose create for new facts, skip for duplicates, or update for a newer or contradictory version. For update, memory_key must be one listed above. Never merge unrelated facts or invent data.""")
    if decision.action == "update" and decision.memory_key not in {item.key for item in matches}:
        return MemoryDecision(action="create", text=candidate)
    return MemoryDecision(action="create", text=candidate) if decision.action == "create" and not decision.text else decision


async def save_long_term_memory(state: dict, config: RunnableConfig):
    user_message = latest_human_text(state.get("messages", []))
    if not user_message or user_message.lower().strip(".!? ") in _IGNORED_MEMORY_TEXT:
        return {}
    namespace, thread_id = ("user_memories", _user_id(config)), str(config["configurable"]["thread_id"])
    for candidate in await _extract_durable_memories(user_message):
        decision = await _decide_memory_action(candidate, await memory_store.asearch(namespace, query=candidate, limit=2))
        if decision.action != "skip":
            value = {"text": decision.text or candidate, "source_thread_id": thread_id, "updated_at": _now()}
            key = decision.memory_key if decision.action == "update" else str(uuid.uuid4())
            await memory_store.aput(namespace, key=key, value=value, index=["text"])
    return {}


async def _retrieve_all_threads(user_id: str) -> list[str]:
    user_id, thread_ids = str(user_id), set()
    metadata = await memory_store.asearch((THREAD_METADATA_NAMESPACE, user_id), limit=100)
    thread_ids.update(str(item.key) for item in metadata)
    async for checkpoint in checkpointer.alist(None):
        configurable = checkpoint.config.get("configurable", {})
        if str(configurable.get("user_id")) == user_id:
            thread_ids.add(str(configurable["thread_id"]))
    return list(thread_ids)


def retrieve_all_threads(user_id: str) -> list[str]:
    return run_async(_retrieve_all_threads(user_id))
