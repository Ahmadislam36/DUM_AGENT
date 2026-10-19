from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: Literal[
        "chat", "daily_update", "generate_report", "query_history", "update_profile",
        "delete_task", "update_task", "delete_profile",
    ] = Field(
        description="The user's intent."
    )


class ExtractedLog(BaseModel):
    date: str = Field(
        default="",
        description="Date of the work in YYYY-MM-DD. Use today's date if the user did not mention one.",
    )
    time_in: Optional[str] = Field(default=None, description="Check-in time, e.g. '09:00'.")
    time_out: Optional[str] = Field(default=None, description="Check-out time, e.g. '17:30'.")
    project_name: str = Field(
        default="", description="Project worked on. Empty string if not stated or ambiguous."
    )
    tasks_completed: List[str] = Field(
        default_factory=list, description="Concrete tasks completed. Empty if none/vague."
    )
    meetings: Optional[str] = Field(
        default=None, description="Meetings attended, with outcomes if mentioned."
    )
    is_clear: bool = Field(
        description="True only if project_name and tasks_completed are both unambiguous."
    )
    missing_info_reason: Optional[str] = Field(
        default=None, description="What is missing or vague when is_clear is False."
    )


class ClarifyQuestions(BaseModel):
    questions: List[str] = Field(description="1 to 3 short, friendly follow-up questions.")


class SearchFilter(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Inclusive start, YYYY-MM-DD.")
    end_date: Optional[str] = Field(default=None, description="Inclusive end, YYYY-MM-DD.")
    keywords: List[str] = Field(
        default_factory=list,
        description="Project/topic/person keywords to match. Empty for purely date-based questions.",
    )


class ProfileInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="The user's name, if stated.")
    role: Optional[str] = Field(default=None, description="Job title/role, e.g. 'AI Engineer'.")
    company: Optional[str] = Field(default=None, description="Company/employer name, if stated.")


class TaskTarget(BaseModel):
    log_id: Optional[int] = Field(default=None, description="Log number if the user says e.g. 'log #5'.")
    date: Optional[str] = Field(default=None, description="Date of the log, YYYY-MM-DD, if mentioned.")
    project_name: Optional[str] = Field(default=None, description="Project name, only if mentioned.")
    task_keyword: Optional[str] = Field(
        default=None, description="Few words identifying the task, e.g. 'CI/CD pipeline'. Not generic words."
    )


class DeleteTarget(TaskTarget):
    delete_whole_log: bool = Field(
        default=False, description="True only if the user wants the entire update/log entry removed."
    )


class EditTarget(TaskTarget):
    new_task_text: Optional[str] = Field(default=None, description="Replacement wording for the task.")
    new_project_name: Optional[str] = Field(default=None, description="New project name, if asked.")
    new_time_in: Optional[str] = Field(default=None, description="New check-in time, HH:MM 24h.")
    new_time_out: Optional[str] = Field(default=None, description="New check-out time, HH:MM 24h.")
    new_meetings: Optional[str] = Field(default=None, description="New meetings text.")
    new_date: Optional[str] = Field(default=None, description="New date, YYYY-MM-DD.")
