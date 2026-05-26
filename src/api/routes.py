"""
API endpoints for the Nexus AI Research Assistant.

Endpoints:
    POST /chat          — Standard request/response chat.
    POST /chat/stream   — Server-Sent Events (SSE) streaming chat.
    POST /new           — Create a new conversation thread.
    GET  /health        — Health check.
"""
import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.api.schemas import ChatRequest, ChatResponse
from src.agent.graph import agent_graph
from src.core.logging import setup_logger

logger = setup_logger("nexus_ai.api")

router = APIRouter()


# ─── Health Check ─────────────────────────────────────────
@router.get("/health", summary="Health check")
async def health():
    """Returns the service health status."""
    return {"status": "ok", "service": "nexus-ai"}


# ─── New Conversation ─────────────────────────────────────
@router.post("/new", summary="Start a new conversation")
async def new_conversation():
    """Creates and returns a fresh thread ID."""
    thread_id = str(uuid.uuid4())
    logger.info(f"New conversation created: {thread_id}")
    return {"thread_id": thread_id}


# ─── Standard Chat ────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse, summary="Process a user query")
async def chat_endpoint(request: ChatRequest):
    """Processes a user query through the multi-agent workflow (non-streaming)."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=request.message)]}

        logger.info(f"[{thread_id[:8]}] Processing query: {request.message[:80]}")

        output_state = None
        agents_used = []
        for chunk in agent_graph.stream(inputs, config, stream_mode="values"):
            output_state = chunk
            active = chunk.get("active_agent")
            if active and active not in agents_used:
                agents_used.append(active)
                logger.info(f"[{thread_id[:8]}] Agent activated: {active}")

        final_message = output_state["messages"][-1].content
        confidence = output_state.get("confidence_score")

        logger.info(f"[{thread_id[:8]}] Complete — agents: {agents_used}, confidence: {confidence}")

        return ChatResponse(
            response=final_message,
            thread_id=thread_id,
            agents_used=agents_used,
            confidence=confidence,
        )
    except Exception as e:
        logger.error(f"[{thread_id[:8]}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── SSE Streaming Chat ──────────────────────────────────
@router.post("/chat/stream", summary="Stream a query via SSE")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Processes a user query and streams results via Server-Sent Events.
    
    Events:
        agent   — Fired when an agent starts processing.
        result  — Fired with the final response payload.
        error   — Fired if an error occurs.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    def event_generator():
        try:
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=request.message)]}

            logger.info(f"[{thread_id[:8]}] SSE stream started: {request.message[:80]}")

            output_state = None
            agents_used = []

            for chunk in agent_graph.stream(inputs, config, stream_mode="values"):
                output_state = chunk
                active = chunk.get("active_agent")
                if active and active not in agents_used:
                    agents_used.append(active)
                    # Send agent activation event
                    yield f"event: agent\ndata: {json.dumps({'agent': active})}\n\n"
                    logger.info(f"[{thread_id[:8]}] SSE agent: {active}")

            final_message = output_state["messages"][-1].content
            confidence = output_state.get("confidence_score")

            # Send final result event
            result = {
                "response": final_message,
                "thread_id": thread_id,
                "agents_used": agents_used,
                "confidence": confidence,
            }
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
            logger.info(f"[{thread_id[:8]}] SSE complete — agents: {agents_used}")

        except Exception as e:
            logger.error(f"[{thread_id[:8]}] SSE error: {str(e)}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
