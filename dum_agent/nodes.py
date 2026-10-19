import re
from datetime import date
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from . import config
from .database import (
    count_logs,
    delete_profile,
    fetch_30_days_logs,
    fetch_all_logs,
    find_profile_by_name,
    get_profile,
    list_profiles,
    save_log,
    search_logs,
    update_profile_fields,
    upsert_profile,
)
from .llm import llm_text, structured
from .logger import get_logger
from .pdf_report import markdown_to_pdf
from .prompts import (
    ANSWER_SYSTEM,
    CLARIFY_SYSTEM,
    CLASSIFIER_SYSTEM,
    DELETE_TARGET_EXTRA,
    EDIT_TARGET_EXTRA,
    EXTRACTOR_SYSTEM,
    PROFILE_SYSTEM,
    REPORT_SYSTEM,
    SEARCH_FILTER_SYSTEM,
    TARGET_BASE,
)
from .schemas import (
    ClarifyQuestions,
    DeleteTarget,
    EditTarget,
    ExtractedLog,
    IntentResult,
    ProfileInfo,
    SearchFilter,
)
from .state import AgentState
from .task_ops import describe, execute_delete, execute_update, find_candidates
from .utils import (
    ACTIVE_PROFILE,
    NO_WORDS,
    YES_WORDS,
    compute_stats,
    draft_raw_input,
    draft_transcript,
    extract_delete_profile_name,
    extract_switch_name,
    last_ai_text,
    last_human_text,
    looks_like_profile_statement,
    logs_to_text,
    normalize_date,
    parse_date,
    quick_intent,
    reply,
    chat_reply_text,
    slugify,
    smalltalk_kind,
    starts_with_word,
    to_json,
    today_context,
)

PROFILE_REQUIRED = {"daily_update", "generate_report", "query_history", "delete_task", "update_task"}
_FINAL_RE = re.compile(r"\b(final|full|complete|overall|all[- ]time)\b", re.I)

log_intent = get_logger("intent")
log_log = get_logger("log")
log_profile = get_logger("profile")
log_report = get_logger("report")
log_search = get_logger("search")
log_task = get_logger("task")


def profile_text(profile) -> str:
    if not profile:
        return ""
    parts = [
        f"Name: {profile['name']}" if profile.get("name") else "",
        f"Role: {profile['role']}" if profile.get("role") else "",
        f"Company: {profile['company']}" if profile.get("company") else "",
    ]
    return ", ".join(p for p in parts if p)


def _pname(profile_id) -> str:
    p = get_profile(profile_id)
    return f"{p['name']} (id={p['id']})" if p else "none"


# ----------------------------- 1. Intent classifier ----------------------------- #
def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    text = last_human_text(messages)
    log_intent.info("Step 1: message received: %r | active profile: %s", text[:80], _pname(state.get("profile_id")))

    # A delete/update flow is waiting for yes/no or a pick: this message is the answer.
    if state.get("pending_action"):
        log_intent.info("Waiting for your answer (%s) -> routing to apply_pending", state["pending_action"].get("type"))
        return {"intent": "apply_pending"}

    awaiting = bool(state.get("awaiting_clarification"))
    if awaiting and text.strip().lower() in config.CANCEL_WORDS:
        log_intent.info("Clarification cancelled by user; draft discarded")
        return reply(
            "No problem - I discarded that draft update.",
            intent="cancel",
            awaiting_clarification=False,
            extracted_log=None,
            clarify_rounds=0,
        )

    source = "command match (no LLM call)"
    if extract_switch_name(text):  # "switch to Wakeel" - free, no LLM
        intent = "switch_profile"
    elif extract_delete_profile_name(text) is not None:  # "delete profile Wakeel" - free, no LLM
        intent = "delete_profile"
    else:
        # Fast path: keyword routing for clear messages (no LLM call). Falls back to the LLM.
        intent = quick_intent(text) if config.FAST_ROUTING else None
        source = "keyword shortcut (no LLM call)"
        if intent is None:
            source = "LLM classifier"
            prompt_text = text
            if awaiting:  # tell the classifier what the assistant just asked
                prompt_text = (
                    "(The assistant just asked the user for missing details about a work update. "
                    f"Its last message was: {last_ai_text(messages)[:300]!r})\nUser message: {text}"
                )
            result = structured(IntentResult).invoke(
                [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=prompt_text)]
            )
            # when unsure: keep the draft going if we were waiting for an answer, otherwise just chat
            intent = result.intent if result else ("daily_update" if awaiting else "chat")
    log_intent.info("Step 2: intent = %s (via %s)", intent, source)

    reset: Dict[str, Any] = {}
    if awaiting:
        if intent == "daily_update":
            log_intent.info("This answers my clarification question -> continue the draft (re-extract with context)")
            return {"intent": "daily_update"}
        # NOT an answer (new request or chat): leave the clarification instead of staying trapped in it
        log_intent.info("This is not an answer to my question -> leaving the clarification, draft discarded")
        reset = {"awaiting_clarification": False, "extracted_log": None, "clarify_rounds": 0}

    if intent in PROFILE_REQUIRED and not state.get("profile_id"):
        log_intent.warning("No active profile -> asking the user to create/select one")
        return reply(
            "No profile selected yet. Create one first, e.g. "
            "\"My name is Ahmad, I work as AI engineer at Abc\", or pick one from the profile list.",
            intent="needs_profile",
            **reset,
        )

    update: Dict[str, Any] = {"intent": intent, **reset}
    if intent == "daily_update":
        update["draft_start"] = len(messages) - 1  # index of the current human message
        update["extracted_log"] = None
        update["is_clear"] = False
        update["clarify_rounds"] = 0
    return update


def route_by_intent(state: AgentState) -> str:
    return {
        "daily_update": "extract_data",
        "generate_report": "fetch_data",
        "query_history": "search_history",
        "update_profile": "update_profile",
        "switch_profile": "switch_profile",
        "delete_profile": "delete_profile",
        "delete_task": "delete_task",
        "update_task": "update_task",
        "apply_pending": "apply_pending",
        "chat": "chat_reply",
    }.get(state.get("intent", ""), END)


# ----------------------------- 2. Daily update branch ----------------------------- #
def extract_data_node(state: AgentState) -> Dict[str, Any]:
    log_log.info("Step 3: extracting date, project, tasks, times, meetings with the LLM")
    system = EXTRACTOR_SYSTEM.format(today=today_context())
    result = structured(ExtractedLog).invoke(
        [SystemMessage(content=system), HumanMessage(content=draft_transcript(state))]
    )

    if result is None:  # rare structured-output failure
        log_log.warning("LLM returned nothing usable -> will ask the user for details")
        log = {
            "date": date.today().isoformat(),
            "time_in": None,
            "time_out": None,
            "project_name": "",
            "tasks_completed": [],
            "meetings": None,
            "is_clear": False,
            "missing_info_reason": "Could not parse the update; project and tasks are needed.",
        }
        return {"extracted_log": log, "is_clear": False}

    log = result.model_dump()
    log["date"] = normalize_date(log.get("date"))
    log["project_name"] = (log.get("project_name") or "").strip()
    log["tasks_completed"] = [t.strip() for t in log.get("tasks_completed", []) if t and t.strip()]

    # Hard guard: never trust is_clear if mandatory fields are empty.
    is_clear = bool(log["is_clear"]) and bool(log["project_name"]) and bool(log["tasks_completed"])
    if not is_clear and not log.get("missing_info_reason"):
        missing = []
        if not log["project_name"]:
            missing.append("project name")
        if not log["tasks_completed"]:
            missing.append("specific tasks completed")
        log["missing_info_reason"] = "Missing: " + ", ".join(missing)
    log["is_clear"] = is_clear
    log_log.info(
        "Extracted: date=%s project=%r tasks=%d time=%s-%s meetings=%s | clear=%s",
        log["date"], log["project_name"], len(log["tasks_completed"]),
        log.get("time_in") or "-", log.get("time_out") or "-", log.get("meetings") or "none", is_clear,
    )
    if not is_clear:
        log_log.info("Not clear enough -> will ask follow-up questions (%s)", log.get("missing_info_reason"))
    return {"extracted_log": log, "is_clear": is_clear}


def route_after_extract(state: AgentState) -> str:
    return "save_to_db" if state.get("is_clear") else "clarify"


def save_to_db_node(state: AgentState) -> Dict[str, Any]:
    log = state["extracted_log"]
    profile = get_profile(state["profile_id"])
    log_log.info("Step 4: saving the update to the database for %s", _pname(state["profile_id"]))
    log_id = save_log(log, draft_raw_input(state), state["profile_id"])
    log_log.info("Step 5: done -> log #%s saved (project=%r, %d task(s))", log_id, log["project_name"], len(log["tasks_completed"]))
    text = (
        f"Saved for {profile['name']}! (log #{log_id})\n"
        f"- Date: {log['date']}\n"
        f"- Project: {log['project_name']}\n"
        f"- Tasks: {len(log['tasks_completed'])} logged\n"
        f"- Time: {log.get('time_in') or '-'} to {log.get('time_out') or '-'}\n"
        f"- Meetings: {log.get('meetings') or 'none'}"
    )
    return reply(text, awaiting_clarification=False, clarify_rounds=0)


def clarify_node(state: AgentState) -> Dict[str, Any]:
    log = state.get("extracted_log") or {}
    reason = log.get("missing_info_reason") or "Project name and specific tasks are needed."
    rounds = state.get("clarify_rounds", 0) + 1
    if rounds > config.MAX_CLARIFY_ROUNDS:
        log_log.warning("Still unclear after %d follow-up round(s) -> giving up, nothing saved", config.MAX_CLARIFY_ROUNDS)
        return reply(
            "I still couldn't pick out the project and the tasks, so I did not save anything. "
            "Try one message like: \"Today I worked on LawGPT and wrote the CI/CD pipeline, 9 to 5\".",
            awaiting_clarification=False,
            extracted_log=None,
            clarify_rounds=0,
        )
    log_log.info("Step 4: generating follow-up questions with the LLM (round %d of %d)", rounds, config.MAX_CLARIFY_ROUNDS)
    result = structured(ClarifyQuestions).invoke(
        [
            SystemMessage(content=CLARIFY_SYSTEM),
            HumanMessage(
                content=f"Conversation so far:\n{draft_transcript(state)}\n\nWhat is missing: {reason}"
            ),
        ]
    )
    questions = [q.strip() for q in (result.questions if result else []) if q and q.strip()][:3]
    if not questions:
        questions = ["Which project did you work on, and what exactly did you complete?"]
    log_log.info("Asked %d follow-up question(s); waiting for your reply (nothing saved yet)", len(questions))
    text = "Almost there - a few quick questions so I can log this properly:\n" + "\n".join(
        f"{i}. {q}" for i, q in enumerate(questions, 1)
    )
    return reply(text, awaiting_clarification=True, clarify_rounds=rounds)


# ----------------------------- 3. Report branch ----------------------------- #
def fetch_data_node(state: AgentState) -> Dict[str, Any]:
    final = bool(_FINAL_RE.search(last_human_text(state["messages"])))
    pid = state["profile_id"]
    logs = fetch_all_logs(pid) if final else fetch_30_days_logs(pid)
    kind = "Final" if final else "Monthly"
    log_report.info(
        "Step 3: %s report requested for %s -> fetched %d log(s) (%s)",
        kind, _pname(pid), len(logs), "all time" if final else "last 30 days",
    )
    return {"logs": logs, "report_kind": kind, "report_path": ""}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs") or []
    kind = state.get("report_kind", "Monthly")
    profile = get_profile(state["profile_id"])
    if not logs:
        log_report.warning("No logs found for %s -> nothing to report", profile["name"])
        scope = "the past 30 days" if kind == "Monthly" else "its history"
        return reply(f"{profile['name']} has no logs in {scope}, so there's nothing to report.", report_path="")

    stats = compute_stats(logs)
    log_report.info(
        "Step 4: stats computed: %d entries, %d day(s), %.1f hours | writing the report with the LLM",
        stats["entries"], stats["days_logged"], stats["total_hours"],
    )
    period = f"{logs[0]['date']} to {date.today().isoformat()}"
    report = llm_text(
        [
            SystemMessage(content=REPORT_SYSTEM),
            HumanMessage(
                content=(
                    f"Report type: {kind}\nPeriod: {period}\n"
                    f"Profile: {profile_text(profile)}\n\n"
                    f"Computed stats (authoritative):\n{to_json(stats)}\n\n"
                    f"Logs:\n{logs_to_text(logs)}"
                )
            ),
        ]
    )

    out_dir = config.REPORTS_DIR / slugify(profile["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{kind.lower()}_report_{date.today().isoformat()}"
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    log_report.info("Step 5: markdown saved -> %s", out_dir / f"{stem}.md")
    try:
        pdf_path = str(markdown_to_pdf(report, out_dir / f"{stem}.pdf", title=f"{kind} Report - {profile['name']}"))
        footer = f"Saved PDF: {pdf_path}"
        log_report.info("Step 6: PDF created -> %s", pdf_path)
    except Exception as e:  # keep the text report even if PDF fails
        pdf_path = ""
        footer = f"(PDF could not be created: {e}) Markdown saved in {out_dir}"
        log_report.error("PDF creation failed: %s: %s", type(e).__name__, e)
    return reply(f"{report}\n\n---\n{footer}", report_path=pdf_path)


# ----------------------------- 4. Query branch ----------------------------- #
def search_history_node(state: AgentState) -> Dict[str, Any]:
    question = last_human_text(state["messages"])
    log_search.info("Step 3: turning the question into a database filter with the LLM")
    flt = structured(SearchFilter).invoke(
        [
            SystemMessage(content=SEARCH_FILTER_SYSTEM.format(today=today_context())),
            HumanMessage(content=question),
        ]
    ) or SearchFilter()
    logs = search_logs(
        profile_id=state["profile_id"],
        start_date=normalize_date(flt.start_date) if flt.start_date else None,
        end_date=normalize_date(flt.end_date) if flt.end_date else None,
        keywords=flt.keywords,
    )
    log_search.info(
        "Step 4: searched %s's logs: dates=%s..%s keywords=%s -> %d log(s) found",
        _pname(state["profile_id"]), flt.start_date or "any", flt.end_date or "any", flt.keywords or "none", len(logs),
    )
    return {"logs": logs}


def format_response_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs") or []
    profile = get_profile(state["profile_id"])
    log_search.info("Step 5: writing the answer from %d log(s) with the LLM", len(logs))
    answer = llm_text(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"{today_context()}\n\nQuestion: {last_human_text(state['messages'])}\n\n"
                    f"Active profile: {profile_text(profile)}\n\n"
                    f"Logs of this profile:\n{logs_to_text(logs) or 'none found'}"
                )
            ),
        ]
    )
    return reply(answer)


# ----------------------------- 5. Profile branches ----------------------------- #
def update_profile_node(state: AgentState) -> Dict[str, Any]:
    text = last_human_text(state["messages"])
    if not looks_like_profile_statement(text):
        log_profile.warning("Message does not look like a statement about your name/role/company -> profile NOT changed")
        return reply(
            "That sounded like a normal message, so I left your profile as it is. To change it, say for example: "
            "\"My company is Abc\" or \"I work as AI engineer at Abc\"."
        )
    log_profile.info("Step 3: extracting name / role / company with the LLM")
    result = structured(ProfileInfo).invoke(
        [SystemMessage(content=PROFILE_SYSTEM), HumanMessage(content=text)]
    )
    info = {
        k: ((getattr(result, k, None) or "").strip() or None) if result else None
        for k in ("name", "role", "company")
    }
    log_profile.info("Extracted: name=%r role=%r company=%r", info["name"], info["role"], info["company"])

    if info["name"]:  # new name -> create profile; known name -> update it. Either way switch to it.
        existing = find_profile_by_name(info["name"])
        log_profile.info(
            "Step 4: profile %r %s -> %s",
            info["name"], "already exists" if existing else "is new", "updating role/company" if existing else "creating it",
        )
        profile, created = upsert_profile(info["name"], info["role"], info["company"])
        verb = "Created" if created else "Updated"
        log_profile.info("Step 5: %s profile %s; it is now the active profile", verb.lower(), _pname(profile["id"]))
        return reply(
            f"{verb} profile - {profile_text(profile)}\nNow working in {profile['name']}'s profile.",
            profile_id=profile["id"],
            awaiting_clarification=False,
            extracted_log=None,
        )

    pid = state.get("profile_id")
    if pid and (info["role"] or info["company"]):  # no name -> update the active profile
        log_profile.info("No name given -> updating the active profile %s", _pname(pid))
        profile = update_profile_fields(pid, info["role"], info["company"])
        return reply(f"Updated profile - {profile_text(profile)}")

    log_profile.warning("No name found in the message -> cannot create or update a profile")
    return reply(
        "I need a name to create or update a profile. "
        "Try: \"My name is Ahmad, I work as AI engineer at Abc company\"."
    )


def switch_profile_node(state: AgentState) -> Dict[str, Any]:
    name = extract_switch_name(last_human_text(state["messages"])) or ""
    log_profile.info("Step 3: switch requested -> looking for profile %r", name)
    profile = find_profile_by_name(name)
    if not profile:
        names = ", ".join(p["name"] for p in list_profiles()) or "none yet"
        log_profile.warning("Profile %r not found (available: %s)", name, names)
        return reply(f"No profile named '{name}'. Available: {names}.")
    log_profile.info("Step 4: switched to %s; updates/questions/reports now use this profile's data", _pname(profile["id"]))
    return reply(
        f"Switched to {profile['name']}'s profile. Updates, questions and reports now use {profile['name']}'s data.",
        profile_id=profile["id"],
        awaiting_clarification=False,
        extracted_log=None,
    )


# ----------------------------- 6. Delete / update branches ----------------------------- #
def delete_profile_node(state: AgentState) -> Dict[str, Any]:
    name = extract_delete_profile_name(last_human_text(state["messages"]))
    if name == ACTIVE_PROFILE or name is None:
        profile = get_profile(state.get("profile_id"))
    else:
        profile = find_profile_by_name(name)
    if not profile:
        names = ", ".join(p["name"] for p in list_profiles()) or "none yet"
        log_profile.warning("Delete requested for %r but no such profile (available: %s)", name, names)
        return reply(f"I couldn't find that profile. Available: {names}. Try: \"delete profile Wakeel\".")
    n = count_logs(profile["id"])
    log_profile.info(
        "Step 3: delete requested for %s (%d log(s)) -> asking for confirmation, nothing deleted yet",
        _pname(profile["id"]), n,
    )
    return reply(
        f"This will permanently delete profile '{profile['name']}' and its {n} log(s). "
        "Saved report files stay on disk. This cannot be undone.\n"
        "Reply 'yes' to confirm or 'cancel'.",
        pending_action={"type": "confirm_delete_profile", "profile_id": profile["id"], "name": profile["name"]},
    )


def _pick_prompt(verb: str, cands, truncated: bool) -> str:
    lines = "\n".join(f"{i}. {describe(c)}" for i, c in enumerate(cands, 1))
    more = "\n(Showing the newest matches only - add a project or date to narrow it down.)" if truncated else ""
    return f"I found {len(cands)} matches:\n{lines}{more}\nReply with the number(s) to {verb} (e.g. 2 or 1,3), or 'cancel'."


def _not_found(profile) -> Dict[str, Any]:
    log_task.warning("No matching task found in %s's logs", profile["name"])
    return reply(
        f"I couldn't find a matching task in {profile['name']}'s logs. "
        "Mention the project, date, or log number (shown as 'log #5' in the Logs tab or after saving)."
    )


def delete_task_node(state: AgentState) -> Dict[str, Any]:
    text = last_human_text(state["messages"])
    profile = get_profile(state["profile_id"])
    log_task.info("Step 3: working out which task to delete (LLM)")
    system = TARGET_BASE.format(verb="delete", today=today_context()) + DELETE_TARGET_EXTRA
    t = structured(DeleteTarget).invoke([SystemMessage(content=system), HumanMessage(content=text)]) or DeleteTarget()
    log_task.info(
        "Target: log_id=%s date=%s project=%r keyword=%r whole_log=%s",
        t.log_id, t.date, t.project_name, t.task_keyword, t.delete_whole_log,
    )

    cands, truncated = find_candidates(
        state["profile_id"], log_id=t.log_id, date=parse_date(t.date), project=t.project_name,
        keyword=t.task_keyword, task_level=not t.delete_whole_log,
    )
    log_task.info("Step 4: %d match(es) in %s's logs", len(cands), profile["name"])
    if not cands:
        return _not_found(profile)
    if len(cands) == 1:
        log_task.info("One match -> asking for confirmation, nothing deleted yet")
        what = "this whole update" if t.delete_whole_log else "this task"
        return reply(
            f"Delete {what} from {profile['name']}'s logs?\n{describe(cands[0])}\nReply 'yes' to confirm or 'cancel'.",
            pending_action={"type": "confirm_delete", "items": cands},
        )
    log_task.info("Several matches -> asking which one(s) to delete")
    return reply(_pick_prompt("delete", cands, truncated), pending_action={"type": "pick", "action": "delete", "items": cands})


def update_task_node(state: AgentState) -> Dict[str, Any]:
    text = last_human_text(state["messages"])
    profile = get_profile(state["profile_id"])
    log_task.info("Step 3: working out which task to change and what to change (LLM)")
    system = TARGET_BASE.format(verb="change", today=today_context()) + EDIT_TARGET_EXTRA
    e = structured(EditTarget).invoke([SystemMessage(content=system), HumanMessage(content=text)]) or EditTarget()

    edit = {
        "new_task_text": (e.new_task_text or "").strip() or None,
        "project_name": (e.new_project_name or "").strip() or None,
        "time_in": (e.new_time_in or "").strip() or None,
        "time_out": (e.new_time_out or "").strip() or None,
        "meetings": (e.new_meetings or "").strip() or None,
        "date": parse_date(e.new_date),
    }
    log_task.info(
        "Target: log_id=%s date=%s project=%r keyword=%r | changes: %s",
        e.log_id, e.date, e.project_name, e.task_keyword, {k: v for k, v in edit.items() if v} or "none",
    )
    if e.new_date and not edit["date"]:
        log_task.warning("Could not read the new date %r", e.new_date)
        return reply("I couldn't read the new date. Please use a format like 2026-10-05.")
    if not any(edit.values()):
        log_task.warning("No change requested -> asking what to change")
        return reply(
            "What should I change? Example: \"update the CI/CD task in lawgpt to: CI/CD with GitHub Actions\" "
            "or \"change log #5 project to Alpha\"."
        )

    cands, truncated = find_candidates(
        state["profile_id"], log_id=e.log_id, date=parse_date(e.date), project=e.project_name,
        keyword=e.task_keyword, task_level=bool(edit["new_task_text"]),
    )
    log_task.info("Step 4: %d match(es) in %s's logs", len(cands), profile["name"])
    if not cands:
        return _not_found(profile)
    if len(cands) == 1:
        log_task.info("One match -> applying the update now")
        lines = execute_update(state["profile_id"], cands, edit)
        log_task.info("Step 5: update done")
        return reply("\n".join(lines))
    log_task.info("Several matches -> asking which one(s) to update")
    return reply(
        _pick_prompt("update", cands, truncated),
        pending_action={"type": "pick", "action": "update", "items": cands, "edit": edit},
    )


def apply_pending_node(state: AgentState) -> Dict[str, Any]:
    pa = state.get("pending_action") or {}
    text = last_human_text(state["messages"]).strip().lower()
    pid = state.get("profile_id")
    kind = pa.get("type")
    log_task.info("Step 3: your answer %r to the pending '%s' question", text[:40], kind)

    if text in config.CANCEL_WORDS or text in NO_WORDS:
        log_task.info("Cancelled by user; nothing was changed")
        return reply("Okay, cancelled. Nothing was changed.", pending_action=None, deleted_profile_id=None)

    if kind in ("confirm_delete_profile", "confirm_delete"):
        if not starts_with_word(text, YES_WORDS):
            log_task.info("Answer was not a clear yes -> asking again")
            return reply("Please reply 'yes' to confirm, or 'cancel'.", deleted_profile_id=None)
        if kind == "confirm_delete":
            log_task.info("Step 4: confirmed -> deleting")
            lines = execute_delete(pid, pa["items"])
            log_task.info("Step 5: delete finished")
            return reply("\n".join(lines), pending_action=None, deleted_profile_id=None)
        log_profile.info("Step 4: confirmed -> deleting profile %r and all its logs", pa["name"])
        n = delete_profile(pa["profile_id"])
        extra: Dict[str, Any] = {}
        if pa["profile_id"] == pid:  # active profile is gone -> fall back to another one (or none)
            remaining = list_profiles()
            extra["profile_id"] = remaining[0]["id"] if remaining else None
            log_profile.info("Active profile was deleted -> now using %s", _pname(extra["profile_id"]))
        log_profile.info("Step 5: profile %r deleted (%d log(s) removed)", pa["name"], n)
        return reply(
            f"Deleted profile '{pa['name']}' and {n} log(s).",
            pending_action=None, deleted_profile_id=pa["profile_id"], **extra,
        )

    if kind == "pick":
        items = pa.get("items", [])
        nums = list(dict.fromkeys(int(n) for n in re.findall(r"\d+", text)))
        chosen = [items[n - 1] for n in nums if 1 <= n <= len(items)]
        if not chosen:
            log_task.info("No valid number in the answer -> asking again")
            return reply("Reply with the number(s) from the list (e.g. 2 or 1,3), or 'cancel'.", deleted_profile_id=None)
        log_task.info("Step 4: you picked %d item(s) -> %s", len(chosen), pa.get("action"))
        if pa.get("action") == "delete":
            lines = execute_delete(pid, chosen)
        else:
            lines = execute_update(pid, chosen, pa.get("edit", {}))
        log_task.info("Step 5: %s finished", pa.get("action"))
        return reply("\n".join(lines), pending_action=None, deleted_profile_id=None)

    return reply("Nothing pending.", pending_action=None, deleted_profile_id=None)


# ----------------------------- 7. Chat branch ----------------------------- #
def chat_reply_node(state: AgentState) -> Dict[str, Any]:
    """Greetings get a short greeting back; anything off-topic is politely declined. No LLM, nothing saved."""
    text = last_human_text(state["messages"])
    profile = get_profile(state.get("profile_id"))
    kind = smalltalk_kind(text)
    log_intent.info(
        "Step 3: not a work-log request -> %s reply, then asking what to do (no LLM call, nothing saved)",
        kind or "off-topic (declined)",
    )
    return reply(chat_reply_text(kind, text, profile["name"] if profile else None))
