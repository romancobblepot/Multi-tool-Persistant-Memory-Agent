from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from crag import rag_tool
from documents import thread_has_document
from mcp_tools import get_stock_price, mcp_tools
from persistence import checkpointer, memory_store, retrieve_long_term_memory, save_long_term_memory
from settings import crag_model, model
from persistence import latest_human_text


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_memories: list[str]
    knowledge_route: Literal["parametric", "retrieval"]


class KnowledgeRoute(BaseModel):
    can_use_parametric_knowledge: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


PARAMETRIC_CONFIDENCE_THRESHOLD = 0.90
knowledge_router_llm = crag_model.with_structured_output(KnowledgeRoute)
tools = [get_stock_price, rag_tool, *mcp_tools]
llm_with_tools = model.bind_tools(tools)


def _system_prompt(memories: list[str]) -> SystemMessage:
    history = "\n".join(f"- {memory}" for memory in memories) or "- No relevant long-term user history was found."
    return SystemMessage(content=f"""You are a helpful, accurate assistant.

Use relevant long-term user history only when it genuinely helps answer the latest message. Do not mention this memory system, invent user facts, or force unrelated details.

Relevant user history:
{history}

Answer the latest user message directly. After your answer, add the heading `Questions you may want to explore:` followed by exactly three concise, useful questions directly relevant to the latest message.""")


def _trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    trimmed = trim_messages(messages, max_tokens=5500, token_counter=model, strategy="last", include_system=False, allow_partial=False, start_on="human")
    if trimmed:
        return trimmed
    latest = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    if latest is None:
        raise ValueError("No HumanMessage found in conversation state.")
    return [latest]


async def knowledge_router(state: ChatState, config: RunnableConfig):
    query = latest_human_text(state.get("messages", []))
    if not query:
        return {"knowledge_route": "retrieval"}
    thread_id = str(config["configurable"]["thread_id"])
    decision = await knowledge_router_llm.ainvoke([
        SystemMessage(content="Decide whether a request can be answered safely using only stable general parametric knowledge. Set can_use_parametric_knowledge=True only for clearly stable, general, non-time-sensitive questions needing no citations, verification, tools, or document retrieval. Always set False for current information, search requests, niche factual claims, medical/legal/financial topics, or references to a PDF, document, file, book, chapter, page, source, citation, or 'this document'. When uncertain, set False. Use confidence >= 0.92 only when direct answering is clearly safe."),
        HumanMessage(content=f"Document attached: {thread_has_document(thread_id)}\n\nUser query:\n{query}"),
    ])
    route = "parametric" if decision.can_use_parametric_knowledge and decision.confidence >= PARAMETRIC_CONFIDENCE_THRESHOLD else "retrieval"
    return {"knowledge_route": route}


async def parametric_chat_node(state: ChatState):
    messages = [_system_prompt(state.get("user_memories", [])), SystemMessage(content="Answer directly using parametric knowledge. Do not use tools, web search, or document retrieval."), *_trim_messages(state["messages"])]
    return {"messages": [await model.ainvoke(messages)]}


async def retrieval_chat_node(state: ChatState):
    messages = [_system_prompt(state.get("user_memories", [])), SystemMessage(content="This request requires retrieval or external verification. For factual information requests, use rag_tool first so Corrective RAG can select document evidence, Tavily web search, or a hybrid result."), *_trim_messages(state["messages"])]
    return {"messages": [await llm_with_tools.ainvoke(messages)]}


def _route_after_router(state: ChatState) -> str:
    return "parametric_chat_node" if state.get("knowledge_route") == "parametric" else "retrieval_chat_node"


def _route_after_retrieval(state: ChatState) -> str:
    message = state["messages"][-1]
    return "tools" if isinstance(message, AIMessage) and message.tool_calls else "save_long_term_memory"


tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node("retrieve_long_term_memory", retrieve_long_term_memory)
graph.add_node("knowledge_router", knowledge_router)
graph.add_node("parametric_chat_node", parametric_chat_node)
graph.add_node("retrieval_chat_node", retrieval_chat_node)
graph.add_node("tools", tool_node)
graph.add_node("save_long_term_memory", save_long_term_memory)
graph.add_edge(START, "retrieve_long_term_memory")
graph.add_edge("retrieve_long_term_memory", "knowledge_router")
graph.add_conditional_edges("knowledge_router", _route_after_router, {"parametric_chat_node": "parametric_chat_node", "retrieval_chat_node": "retrieval_chat_node"})
graph.add_edge("parametric_chat_node", "save_long_term_memory")
graph.add_conditional_edges("retrieval_chat_node", _route_after_retrieval, {"tools": "tools", "save_long_term_memory": "save_long_term_memory"})
graph.add_edge("tools", "retrieval_chat_node")
graph.add_edge("save_long_term_memory", END)
chatbot = graph.compile(checkpointer=checkpointer, store=memory_store)

