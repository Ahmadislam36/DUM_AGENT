"""Find / delete / update specific tasks inside one profile's logs."""
from typing import Any, Dict, List, Optional, Tuple

from .database import fetch_all_logs, remove_task, delete_log, update_log


def find_candidates(
    profile_id: int,
    log_id: Optional[int] = None,
    date: Optional[str] = None,
    project: Optional[str] = None,
    keyword: Optional[str] = None,
    task_level: bool = True,
    limit: int = 15,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Newest-first matches. task_level=True -> one candidate per task; False -> one per log entry."""
    logs = fetch_all_logs(profile_id)
    if log_id is not None:
        logs = [l for l in logs if l["id"] == log_id]
    if date:
        logs = [l for l in logs if l["date"] == date]
    if project:
        logs = [l for l in logs if project.lower() in l["project_name"].lower()]

    words = (keyword or "").lower().split()
    cands: List[Dict[str, Any]] = []
    for l in reversed(logs):
        base = {"log_id": l["id"], "date": l["date"], "project": l["project_name"]}
        if task_level:
            for i, t in enumerate(l["tasks"]):
                if all(w in t.lower() for w in words):
                    cands.append({**base, "task_index": i, "task": t})
        else:
            cands.append({**base, "task_index": None, "task": "; ".join(l["tasks"])})
    return cands[:limit], len(cands) > limit


def describe(c: Dict[str, Any]) -> str:
    kind = "whole update: " if c["task_index"] is None else ""
    return f"[log #{c['log_id']}] {c['date']} | {c['project']} | {kind}{c['task']}"


def execute_delete(profile_id: int, items: List[Dict[str, Any]]) -> List[str]:
    lines = []
    # delete higher task indexes first so earlier indexes in the same log stay valid
    for c in sorted(items, key=lambda c: (c["log_id"], -(c["task_index"] if c["task_index"] is not None else -1))):
        if c["task_index"] is None:
            ok = delete_log(profile_id, c["log_id"])
            lines.append(("Deleted " if ok else "Not found: ") + describe(c))
        else:
            res = remove_task(profile_id, c["log_id"], c["task_index"], expected_text=c["task"])
            if res == "not_found":
                lines.append("Not found (already changed?): " + describe(c))
            else:
                extra = " (it was the last task, so the whole update was removed)" if res == "log_deleted" else ""
                lines.append("Deleted " + describe(c) + extra)
    return lines


def execute_update(profile_id: int, items: List[Dict[str, Any]], edit: Dict[str, Any]) -> List[str]:
    lines = []
    for c in items:
        updated = update_log(
            profile_id,
            c["log_id"],
            task_index=c["task_index"],
            new_task_text=edit.get("new_task_text"),
            expected_text=c["task"] if c["task_index"] is not None else None,
            project_name=edit.get("project_name"),
            time_in=edit.get("time_in"),
            time_out=edit.get("time_out"),
            meetings=edit.get("meetings"),
            date=edit.get("date"),
        )
        if not updated:
            lines.append("Not found (already changed?): " + describe(c))
            continue
        changes = []
        if edit.get("new_task_text"):
            changes.append(f"task: '{c['task']}' -> '{edit['new_task_text']}'")
        for key, label in (("project_name", "project"), ("date", "date"), ("time_in", "in"),
                           ("time_out", "out"), ("meetings", "meetings")):
            if edit.get(key):
                changes.append(f"{label} -> {edit[key]}")
        lines.append(f"Updated log #{c['log_id']} ({updated['project_name']}): " + "; ".join(changes))
    return lines
