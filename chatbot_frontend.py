import json
import queue
import uuid
from pathlib import Path
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from chatbot import (chatbot,clear_thread_document,create_thread_metadata, create_user_profile,ensure_user_profile,generate_and_save_thread_title,ingest_pdf,
list_thread_metadata,list_user_profiles,retrieve_all_threads,submit_async_task,thread_document_metadata)


def generate_thread_id():
    return str(uuid.uuid4())

def get_user_threads(user_id=None):
    """Return only the conversations owned by one UUID-based user profile."""
    user_id = str(user_id or st.session_state["user_id"])
    threads_by_user = st.session_state["chat_threads_by_user"]
    if user_id not in threads_by_user:
        threads_by_user[user_id] = retrieve_all_threads(user_id)
    return threads_by_user[user_id]


def refresh_user_profiles():
    """Load persistent pre-auth profiles from Supabase for the switcher."""
    profiles = list_user_profiles()
    st.session_state["user_profiles"] = {
        profile["user_id"]: profile for profile in profiles
    }
    return profiles


def refresh_thread_metadata(user_id=None):
    user_id = str(user_id or st.session_state["user_id"])
    metadata = list_thread_metadata(user_id)
    st.session_state["thread_metadata_by_user"][user_id] = metadata
    return metadata


def ensure_thread_metadata(thread_id, user_id=None):
    user_id = str(user_id or st.session_state["user_id"])
    metadata = st.session_state["thread_metadata_by_user"].setdefault(user_id, {})
    if str(thread_id) not in metadata:
        metadata[str(thread_id)] = create_thread_metadata(user_id, str(thread_id))
    return metadata[str(thread_id)]


def thread_title(thread_id):
    metadata = ensure_thread_metadata(thread_id)
    return metadata.get("title") or "New chat"


def schedule_thread_title():
    user_id, thread_id = st.session_state["user_id"], str(st.session_state["thread_id"])
    metadata = ensure_thread_metadata(thread_id, user_id)
    if metadata.get("title_source") in {"auto", "manual"}:
        return
    first_prompt = next((item["content"] for item in st.session_state["message_history"] if item["role"] == "user"), "")
    task_key = f"{user_id}:{thread_id}"
    if first_prompt and task_key not in st.session_state["title_generation_started"]:
        st.session_state["title_generation_started"].add(task_key)
        submit_async_task(generate_and_save_thread_title(user_id, thread_id, first_prompt))


def is_response_chunk(chunk, metadata):
    return (
        isinstance(chunk, AIMessageChunk)
        and metadata.get("langgraph_node") in {"chat_node", "parametric_chat_node", "retrieval_chat_node"}
        and isinstance(chunk.content, str)
        and chunk.content
    )


def add_thread(thread_id):
    threads = get_user_threads()
    if thread_id not in threads:
        threads.append(thread_id)
    ensure_thread_metadata(thread_id)
    st.session_state["active_thread_by_user"][st.session_state["user_id"]] = thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state.pop("pending_rag_interrupt", None)
    add_thread(thread_id)


def activate_user_profile(user_id):
    """Switch the visible chat state to another in-session user profile."""
    user_id = str(user_id)
    st.session_state["user_id"] = user_id
    st.session_state.pop("pending_rag_interrupt", None)
    threads = get_user_threads(user_id)
    active_thread = st.session_state["active_thread_by_user"].get(user_id)

    if active_thread not in threads:
        if threads:
            active_thread = threads[-1]
        else:
            active_thread = generate_thread_id()
            add_thread(active_thread)
        st.session_state["active_thread_by_user"][user_id] = active_thread

    refresh_thread_metadata(user_id)
    st.session_state["thread_id"] = active_thread
    st.session_state["message_history"] = load_conversation(active_thread)


def find_media_paths(content):
    """Find media paths returned by a Manim tool."""
    valid_extensions = (".mp4", ".gif", ".png")

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return [
                token.strip("\"' ,[]()")
                for token in content.split()
                if token.lower().split("?")[0].endswith(valid_extensions)
            ]
    if isinstance(content, dict):
        return [path for value in content.values() for path in find_media_paths(value)]
    if isinstance(content, list):
        return [path for value in content for path in find_media_paths(value)]
    return []


def extract_sources(tool_content):
    """Extract source/page/score from a RAG tool's JSON result.

    The function accepts common formats such as {"sources": [...]},
    {"documents": [...]}, or a plain list of retrieved documents.
    """
    if isinstance(tool_content, str):
        try:
            tool_content = json.loads(tool_content)
        except json.JSONDecodeError:
            return []

    if isinstance(tool_content, dict):
        candidates = (
            tool_content.get("sources")
            or tool_content.get("documents")
            or tool_content.get("results")
            or tool_content.get("chunks")
            or []
        )
    elif isinstance(tool_content, list):
        candidates = tool_content
    else:
        return []

    sources = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata", {}) or {}
        source = (
            item.get("source")
            or metadata.get("source")
            or metadata.get("filename")
            or metadata.get("file_name")
        )
        page = item.get("page", metadata.get("page", metadata.get("page_number")))
        score = item.get("similarity", item.get("score", item.get("relevance_score")))

        if source is not None or page is not None:
            sources.append({"source": source or "Unknown document", "page": page, "score": score})

    # Keep citations readable if the retriever returns the same chunk repeatedly.
    unique_sources = []
    seen = set()
    for item in sources:
        key = (item["source"], item["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(item)
    return unique_sources


def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources used"):
        for source in sources:
            label = source["source"]
            if source["page"] is not None:
                label += f" — page {source['page']}"
            if isinstance(source["score"], (int, float)):
                label += f" (score: {source['score']:.3f})"
            st.markdown(f"- {label}")


def load_conversation(thread_id):
    """Load only user messages and final answers; hide agent internals."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id,"user_id":st.session_state["user_id"]}})
    messages = state.values.get("messages", [])
    conversation = []
    pending_sources = []

    for message in messages:
        if isinstance(message, HumanMessage):
            conversation.append({"role": "user", "content": message.content, "sources": []})
        elif isinstance(message, ToolMessage):
            pending_sources.extend(extract_sources(message.content))
        elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
            conversation.append(
                {"role": "assistant", "content": message.content, "sources": pending_sources}
            )
            pending_sources = []
    return conversation


# ---------- Session setup ----------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "chat_threads_by_user" not in st.session_state:
    st.session_state["chat_threads_by_user"] = {}
if "active_thread_by_user" not in st.session_state:
    st.session_state["active_thread_by_user"] = {}
if "thread_metadata_by_user" not in st.session_state:
    st.session_state["thread_metadata_by_user"] = {}
if "title_generation_started" not in st.session_state:
    st.session_state["title_generation_started"] = set()
if "user_id" not in st.session_state:
    existing_profiles = refresh_user_profiles()
    if existing_profiles:
        # Pre-auth mode: make the most recently used stored profile selectable
        # immediately. Auth will later provide this identity instead.
        st.session_state["user_id"] = existing_profiles[0]["user_id"]
    else:
        profile = create_user_profile("User 1")
        st.session_state["user_id"] = profile["user_id"]
else:
    # Preserves a profile created by the earlier in-session implementation.
    ensure_user_profile(st.session_state["user_id"])
refresh_user_profiles()
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])
refresh_thread_metadata()
st.session_state["active_thread_by_user"][st.session_state["user_id"]] = (
    st.session_state["thread_id"]
)
thread_key = str(st.session_state["thread_id"])


# ---------- Sidebar ----------
st.sidebar.title("LangGraph RAG Chatbot")

st.sidebar.subheader("User profile")
profile_ids = list(st.session_state["user_profiles"])
selected_user_id = st.sidebar.selectbox(
    "Active profile",
    options=profile_ids,
    index=profile_ids.index(st.session_state["user_id"]),
    format_func=lambda profile_id: st.session_state["user_profiles"][profile_id][
        "display_name"
    ],
)
if selected_user_id != st.session_state["user_id"]:
    activate_user_profile(selected_user_id)
    st.rerun()

new_profile_name = st.sidebar.text_input(
    "New profile name", placeholder="e.g. Adnan", key="new_profile_name"
)
if st.sidebar.button("Create profile", use_container_width=True):
    profile_name = new_profile_name.strip() or (
        f"User {len(st.session_state['user_profiles']) + 1}"
    )
    profile = create_user_profile(profile_name)
    new_user_id = profile["user_id"]
    refresh_user_profiles()
    st.session_state["chat_threads_by_user"][new_user_id] = []
    st.session_state["active_thread_by_user"][new_user_id] = generate_thread_id()
    st.session_state["thread_metadata_by_user"][new_user_id] = {}
    activate_user_profile(new_user_id)
    st.rerun()

st.sidebar.caption("Profiles are saved in Supabase. Authentication will secure identity later.")

if st.sidebar.button("New chat", use_container_width=True):
    reset_chat()
    st.rerun()

doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    st.sidebar.success(
        f"Using {doc_meta.get('filename', 'document')} · "
        f"{doc_meta.get('chunks', 0)} chunks · {doc_meta.get('documents', 0)} pages"
    )
else:
    st.sidebar.info("Upload a PDF to ask document-grounded questions.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"], key=f"upload_{thread_key}"
)
attached_document_key = f"attached_document_{thread_key}"

if uploaded_pdf:
    processed_key = f"processed_{thread_key}_{uploaded_pdf.name}"
    if not st.session_state.get(processed_key):
        with st.sidebar.status("Indexing PDF…", expanded=True) as status:
            ingest_pdf(
                uploaded_pdf.getvalue(), thread_id=thread_key, filename=uploaded_pdf.name
            )
            st.session_state[processed_key] = True
            st.session_state[attached_document_key] = uploaded_pdf.name
            status.update(label="PDF indexed", state="complete", expanded=False)
        st.rerun()
    else:
        st.sidebar.caption(f"{uploaded_pdf.name} is already indexed for this chat.")
elif st.session_state.get(attached_document_key):
    # The uploader's clear (×) button changed its value to None. Remove the
    # corresponding FAISS retriever and all metadata before the next question.
    removed_filename = st.session_state[attached_document_key]
    clear_thread_document(thread_key)
    del st.session_state[attached_document_key]
    st.session_state.pop(f"processed_{thread_key}_{removed_filename}", None)
    st.sidebar.success(f"Removed {removed_filename} and its index.")
    st.rerun()

st.sidebar.header("My conversations")
for saved_thread_id in reversed(get_user_threads()):
    active = saved_thread_id == st.session_state["thread_id"]
    if st.sidebar.button(
        thread_title(saved_thread_id), key=f"thread_{saved_thread_id}", disabled=active
    ):
        st.session_state["thread_id"] = saved_thread_id
        st.session_state["active_thread_by_user"][st.session_state["user_id"]] = (
            saved_thread_id
        )
        st.session_state["message_history"] = load_conversation(saved_thread_id)
        st.rerun()


# ---------- Existing messages ----------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


# ---------- Current turn ----------
pending_interrupt = st.session_state.get("pending_rag_interrupt")
if pending_interrupt:
    with st.chat_message("assistant"):
        st.markdown(f"{pending_interrupt['question']}\n\nReply **yes** or **no** below.")

user_input = st.chat_input(
    "Reply yes or no" if pending_interrupt else "Ask a question about your PDF, or anything else"
)

# A paused LangGraph run must be resumed with Command(resume=...), not with a
# new HumanMessage. The user's normal chat box supplies that resume value.
if pending_interrupt and user_input:
    choice = user_input.strip().lower().rstrip(".!?")

    with st.chat_message("user"):
        st.markdown(user_input)

    if choice not in {"yes", "y", "no", "n"}:
        st.warning("Please reply with exactly **yes** or **no**.")
    else:
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input, "sources": []}
        )
        resume_config = {
            "configurable": {"thread_id": thread_key,"user_id":st.session_state["user_id"]},
            "metadata": {"thread_id": thread_key, "user_id": st.session_state["user_id"]},
        }
        use_web_search = choice in {"yes", "y"}
        resume_sources = []

        with st.chat_message("assistant"):
            status = st.status(
                "🔧 Using `search_tool`…" if use_web_search else "Continuing…",
                expanded=True,
            )

            def resume_stream():
                event_queue = queue.Queue()

                async def run_resume_stream():
                    try:
                        async for mode, payload in chatbot.astream(
                            Command(resume={"use_web_search": use_web_search}),
                            config=resume_config,
                            stream_mode=["messages", "updates"],
                        ):
                            if mode == "messages":
                                chunk, metadata = payload
                                if is_response_chunk(chunk, metadata):
                                    event_queue.put(("answer_token", chunk.content))
                            elif mode == "updates":
                                for _, node_update in payload.items():
                                    if not node_update:
                                        continue
                                    for message in node_update.get("messages", []):
                                        if isinstance(message, AIMessage) and message.tool_calls:
                                            for tool_call in message.tool_calls:
                                                event_queue.put(
                                                    ("tool_start", tool_call.get("name", "tool"))
                                                )
                                        elif isinstance(message, ToolMessage):
                                            tool_name = getattr(message, "name", None) or "tool"
                                            event_queue.put(
                                                ("tool_result", (tool_name, message.content))
                                            )
                    except Exception as exc:
                        event_queue.put(("error", exc))
                    finally:
                        event_queue.put(None)

                submit_async_task(run_resume_stream())

                while True:
                    item = event_queue.get()
                    if item is None:
                        break
                    event_type, data = item
                    if event_type == "error":
                        raise data
                    if event_type == "tool_start":
                        status.update(label=f"🔧 Using `{data}`…", state="running")
                    elif event_type == "tool_result":
                        _, tool_content = data
                        resume_sources.extend(extract_sources(tool_content))
                    elif event_type == "answer_token":
                        yield data

            final_message = st.write_stream(resume_stream())
            status.update(label="✅ Tool finished", state="complete", expanded=False)
            render_sources(resume_sources)

        st.session_state["message_history"].append(
            {"role": "assistant", "content": final_message, "sources": resume_sources}
        )
        schedule_thread_title()
        del st.session_state["pending_rag_interrupt"]
        st.rerun()

if user_input and not pending_interrupt:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key,"user_id":st.session_state["user_id"]},
        "metadata": {"thread_id": thread_key, "user_id": st.session_state["user_id"]},
        "run_name": "chat_turn",
    }
    retrieved_sources = []
    rendered_media = []
    turn_was_interrupted = {"value": False}

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        rag_result_holder = st.empty()

        def ai_only_stream():
            event_queue = queue.Queue()

            async def run_stream():
                buffered_tokens = []
                tool_was_called = False

                try:
                    async for mode, payload in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode=["messages", "updates"],
                    ):
                        if mode == "messages":
                            chunk, metadata = payload
                            is_model_chunk = is_response_chunk(chunk, metadata)
                            if is_model_chunk:
                                # Do not show the first model pass yet: it may
                                # turn out to be a tool-call message. Once a
                                # tool has completed, these are genuine final
                                # response tokens and are yielded immediately.
                                if tool_was_called or metadata.get("langgraph_node") == "parametric_chat_node":
                                    event_queue.put(("answer_token", chunk.content))
                                else:
                                    buffered_tokens.append(chunk.content)

                        elif mode == "updates":
                            if "__interrupt__" in payload:
                                interrupt_data = payload["__interrupt__"][0].value
                                event_queue.put(("rag_interrupt", interrupt_data))
                                continue
                            for _, node_update in payload.items():
                                if not node_update:
                                    continue
                                for message in node_update.get("messages", []):
                                    if isinstance(message, AIMessage) and message.tool_calls:
                                        tool_was_called = True
                                        buffered_tokens.clear()
                                        for tool_call in message.tool_calls:
                                            event_queue.put(("tool_start", tool_call.get("name", "tool")))
                                    elif isinstance(message, ToolMessage):
                                        tool_was_called = True
                                        tool_name = getattr(message, "name", None) or "tool"
                                        event_queue.put(("tool_result", (tool_name, message.content)))
                                    elif isinstance(message, AIMessage) and message.content:
                                        # A no-tool answer was held briefly only to
                                        # determine whether this pass called tools.
                                        if not tool_was_called:
                                            for token in buffered_tokens:
                                                event_queue.put(("answer_token", token))
                                            buffered_tokens.clear()
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                event_type, data = item
                if event_type == "error":
                    raise data
                if event_type == "tool_start":
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"Using `{data}`…", expanded=True)
                    else:
                        status_holder["box"].update(label=f"Using `{data}`…", state="running")
                elif event_type == "tool_result":
                    tool_name, tool_content = data
                    retrieved_sources.extend(extract_sources(tool_content))

                    # ToolNode returns one completed ToolMessage. Stream its
                    # text to the UI as soon as it arrives, before the LLM's
                    # final answer is generated.
                    if tool_name == "rag_tool":
                        with rag_result_holder.container():
                            with st.expander("RAG tool response", expanded=False):
                                try:
                                    st.json(
                                        json.loads(tool_content)
                                        if isinstance(tool_content, str)
                                        else tool_content
                                    )
                                except (TypeError, json.JSONDecodeError):
                                    st.code(str(tool_content))

                    if "manim" in tool_name.lower():
                        rendered_media.extend(find_media_paths(tool_content))
                elif event_type == "rag_interrupt":
                    st.session_state["pending_rag_interrupt"] = data
                    turn_was_interrupted["value"] = True
                    return
                elif event_type == "answer_token":
                    yield data

        ai_message = st.write_stream(ai_only_stream())

        # The pending-interrupt UI is rendered above this chat turn. Restart
        # immediately so Streamlit renders it in the same user interaction.
        if turn_was_interrupted["value"]:
            st.rerun()

        # De-duplicate sources collected during this turn.
        retrieved_sources = [
            item for index, item in enumerate(retrieved_sources)
            if (item["source"], item["page"]) not in {
                (previous["source"], previous["page"])
                for previous in retrieved_sources[:index]
            }
        ]
        render_sources(retrieved_sources)

        for media_path in dict.fromkeys(rendered_media):
            if Path(media_path).exists():
                if media_path.lower().endswith((".mp4", ".gif")):
                    st.video(media_path)
                else:
                    st.image(media_path)
            else:
                st.warning(f"Generated media is not accessible: `{media_path}`")

        if status_holder["box"] is not None:
            status_holder["box"].update(label="Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "sources": retrieved_sources}
    )
    schedule_thread_title()
