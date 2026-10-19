from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    apply_pending_node,
    chat_reply_node,
    classify_intent_node,
    clarify_node,
    delete_profile_node,
    delete_task_node,
    extract_data_node,
    fetch_data_node,
    format_response_node,
    generate_report_node,
    route_after_extract,
    route_by_intent,
    save_to_db_node,
    search_history_node,
    switch_profile_node,
    update_profile_node,
    update_task_node,
)
from .state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent_node)
    g.add_node("extract_data", extract_data_node)
    g.add_node("save_to_db", save_to_db_node)
    g.add_node("clarify", clarify_node)
    g.add_node("fetch_data", fetch_data_node)
    g.add_node("generate_report", generate_report_node)
    g.add_node("search_history", search_history_node)
    g.add_node("format_response", format_response_node)
    g.add_node("update_profile", update_profile_node)
    g.add_node("switch_profile", switch_profile_node)
    g.add_node("delete_profile", delete_profile_node)
    g.add_node("delete_task", delete_task_node)
    g.add_node("update_task", update_task_node)
    g.add_node("apply_pending", apply_pending_node)
    g.add_node("chat_reply", chat_reply_node)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "extract_data": "extract_data",
            "fetch_data": "fetch_data",
            "search_history": "search_history",
            "update_profile": "update_profile",
            "switch_profile": "switch_profile",
            "delete_profile": "delete_profile",
            "delete_task": "delete_task",
            "update_task": "update_task",
            "apply_pending": "apply_pending",
            "chat_reply": "chat_reply",
            END: END,
        },
    )

    # Daily update branch
    g.add_conditional_edges(
        "extract_data", route_after_extract, {"save_to_db": "save_to_db", "clarify": "clarify"}
    )
    g.add_edge("save_to_db", END)
    g.add_edge("clarify", END)  # user's reply re-enters via classify_intent -> extract_data

    # Report branch
    g.add_edge("fetch_data", "generate_report")
    g.add_edge("generate_report", END)

    # Query branch
    g.add_edge("search_history", "format_response")
    g.add_edge("format_response", END)

    # Profile branch
    g.add_edge("update_profile", END)
    g.add_edge("switch_profile", END)

    # Delete / update branches (confirmation happens on the next user turn via apply_pending)
    for node in ("delete_profile", "delete_task", "update_task", "apply_pending", "chat_reply"):
        g.add_edge(node, END)

    return g.compile(checkpointer=MemorySaver())
