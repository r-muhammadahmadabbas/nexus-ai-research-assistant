"""
LangGraph workflow definition for the multi-agent research system.
"""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import (
    clarity_node,
    research_node,
    validator_node,
    synthesis_node,
    ask_clarification,
)
from src.core.config import CONFIDENCE_THRESHOLD, MAX_RESEARCH_ATTEMPTS


def route_after_clarity(state: AgentState) -> str:
    """Routes to clarification interrupt if query is ambiguous, else begins research."""
    if state.get("clarity_status") == "needs_clarification":
        return "ask_clarification"
    return "research_node"


def route_after_research(state: AgentState) -> str:
    """Routes to validation if confidence is low, else skips directly to synthesis."""
    if state.get("confidence_score", 0) < CONFIDENCE_THRESHOLD:
        return "validator_node"
    return "synthesis_node"


def route_after_validator(state: AgentState) -> str:
    """Loops back to research if data is insufficient, preventing infinite loops via attempt counter."""
    is_insufficient = state.get("validation_result") == "insufficient"
    under_max_attempts = state.get("attempts", 0) < MAX_RESEARCH_ATTEMPTS
    if is_insufficient and under_max_attempts:
        return "research_node"
    return "synthesis_node"


def build_graph():
    """Assembles the state machine, mapping nodes to functions and defining transition edges."""
    workflow = StateGraph(AgentState)

    # Register all processing nodes
    workflow.add_node("clarity_node", clarity_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("validator_node", validator_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("ask_clarification", ask_clarification)

    # Define the starting point
    workflow.add_edge(START, "clarity_node")

    # Wire conditional transitions
    workflow.add_conditional_edges(
        "clarity_node",
        route_after_clarity,
        {"ask_clarification": "ask_clarification", "research_node": "research_node"},
    )

    workflow.add_edge("ask_clarification", END)

    workflow.add_conditional_edges(
        "research_node",
        route_after_research,
        {"validator_node": "validator_node", "synthesis_node": "synthesis_node"},
    )

    workflow.add_conditional_edges(
        "validator_node",
        route_after_validator,
        {"research_node": "research_node", "synthesis_node": "synthesis_node"},
    )

    workflow.add_edge("synthesis_node", END)

    # Attach SQLite-backed memory for multi-turn persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

agent_graph = build_graph()
