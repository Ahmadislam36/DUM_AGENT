from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    extracted_log: Optional[Dict[str, Any]]
    is_clear: bool
    response: str
    # internal bookkeeping
    profile_id: Optional[int]  # active profile: every log/query/report is scoped to it
    awaiting_clarification: bool  # True while a clarify round-trip is pending
    clarify_rounds: int  # follow-up rounds already asked for the current draft
    draft_start: int  # index in `messages` where the current draft update began
    logs: List[Dict[str, Any]]  # rows fetched from SQLite for report/query branches
    report_kind: str  # "Monthly" or "Final"
    report_path: str  # PDF path of the last generated report ("" if none)
    pending_action: Optional[Dict[str, Any]]  # waiting for yes/no or a pick (delete/update flows)
    deleted_profile_id: Optional[int]  # set when a profile was just deleted
