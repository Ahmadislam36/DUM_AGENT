import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .state import AgentState


def normalize_date(value: Optional[str]) -> str:
    if value:
        v = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y"):
            try:
                return datetime.strptime(v, fmt).date().isoformat()
            except ValueError:
                continue
    return date.today().isoformat()


def parse_time_minutes(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    s = re.sub(r"\s+", " ", value.strip().lower().replace(".", ""))
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            t = datetime.strptime(s, fmt)
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    return None


def hours_worked(log: Dict[str, Any]) -> Optional[float]:
    t_in, t_out = parse_time_minutes(log.get("time_in")), parse_time_minutes(log.get("time_out"))
    if t_in is None or t_out is None or t_out <= t_in:
        return None
    return round((t_out - t_in) / 60, 2)


def compute_stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total, by_project, by_day, missing = 0.0, {}, {}, 0
    for log in logs:
        h = hours_worked(log)
        if h is None:
            missing += 1
            continue
        total += h
        by_project[log["project_name"]] = round(by_project.get(log["project_name"], 0) + h, 2)
        by_day[log["date"]] = round(by_day.get(log["date"], 0) + h, 2)
    return {
        "entries": len(logs),
        "days_logged": len({l["date"] for l in logs}),
        "total_hours": round(total, 2),
        "hours_by_project": by_project,
        "hours_by_day": by_day,
        "entries_without_valid_times": missing,
    }


def logs_to_text(logs: List[Dict[str, Any]]) -> str:
    out = []
    for l in logs:
        d = datetime.strptime(l["date"], "%Y-%m-%d").strftime("%a") if l.get("date") else ""
        out.append(
            f"- {l['date']} ({d}) | project: {l['project_name']} | "
            f"in: {l.get('time_in') or 'n/a'} | out: {l.get('time_out') or 'n/a'} | "
            f"tasks: {'; '.join(l['tasks'])} | meetings: {l.get('meetings') or 'none'}"
        )
    return "\n".join(out)


def last_human_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def draft_transcript(state: AgentState) -> str:
    msgs = state["messages"][state.get("draft_start", 0):]
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}" for m in msgs
    )


def draft_raw_input(state: AgentState) -> str:
    msgs = state["messages"][state.get("draft_start", 0):]
    return "\n".join(str(m.content) for m in msgs if isinstance(m, HumanMessage))


def today_context() -> str:
    t = date.today()
    return f"Today is {t.strftime('%A')}, {t.isoformat()}."


def reply(text: str, **extra) -> Dict[str, Any]:
    return {"response": text, "messages": [AIMessage(content=text)], **extra}


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2)


# --------------------------- fast intent routing --------------------------- #
_PROFILE_RE = re.compile(
    r"\b(my name is|mera naam|call me|my (?:role|job|job title|title|company|designation) is|"
    r"i work (?:as|at|for)|i am working (?:as|at|for))\b",
    re.I,
)
_PROFILE_GUARD_RE = re.compile(
    _PROFILE_RE.pattern
    + r"|\bi(?:'m| am) an? [\w\s/&-]{2,40}? (?:at|for|in) \w+"
    + r"|\bi(?:'m| am) [\w\s/&-]{2,40}? (?:engineer|developer|manager|designer|analyst|intern|lead|consultant)\b",
    re.I,
)
# A question is only about the user's own logs if it mentions something personal / work-log related
_CUE_RE = re.compile(
    r"\b(my|mera|meri|mere|maine|mujhe|logs?|tasks?|projects?|worked|updates?|week|month|yesterday|"
    r"today|kal|aaj|last|company|role|profile|hours|meetings?|did i|do i|have i|was i|am i)\b",
    re.I,
)
_GREET_START_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|salam|salaam|assalam\S*|aoa|good (?:morning|afternoon|evening|night))\b", re.I
)
_THANKS_RE = re.compile(r"\b(?:thanks|thank you|thx|shukriya|jazak\w*)\b", re.I)
_HOWARE_RE = re.compile(r"\b(?:how are you|how r u|kaise ho|kese ho|kya haal)\b", re.I)
_HELP_RE = re.compile(r"\b(?:what can you do|who are you|what do you do)\b|^\s*(?:help|menu|options)\s*[!.?]*$", re.I)
_TALK_RE = re.compile(r"\b(?:talk|chat|baat)\b.{0,20}\bwith me\b|\bsimple conversation\b", re.I)
_ACK_RE = re.compile(r"^\s*(?:ok|okay|k|theek hai|thik hai|fine|cool|nice)[\s!.]*$", re.I)


def smalltalk_kind(text: str) -> Optional[str]:
    """greeting / thanks / howare / help / talk / ack - or None if it is not small talk."""
    t = text.strip()
    n = len(re.findall(r"\w+", t))
    if _HELP_RE.search(t):
        return "help"
    if _HOWARE_RE.search(t):
        return "howare"
    if _TALK_RE.search(t):
        return "talk"
    if _THANKS_RE.search(t) and n <= 5:
        return "thanks"
    if _ACK_RE.match(t):
        return "ack"
    if _GREET_START_RE.match(t) and n <= 4:
        return "greeting"
    return None


MENU = (
    "What would you like to do today?\n"
    "- Log an update, e.g. `Today I worked on LawGPT and wrote the CI/CD pipeline, 9 to 5`\n"
    "- Ask about your logs, e.g. `What did I do last Tuesday?`\n"
    "- Get a report: `generate my monthly report`\n"
    "- Edit or delete a log, or manage profiles"
)


def chat_reply_text(kind: Optional[str], text: str, name: Optional[str]) -> str:
    """Fixed replies (no LLM): answer the greeting only, then ask about work. Never answers off-topic questions."""
    who = f" {name}" if name else ""
    if kind == "greeting":
        salam = re.search(r"salam|salaam|assalam|aoa", text, re.I)
        return f"{'Walaikum assalam' if salam else 'Hello'}{who}! {MENU}"
    if kind == "howare":
        return f"I'm doing well, thank you! {MENU}"
    if kind == "thanks":
        return "You're welcome! Anything else for your work log today?"
    if kind == "ack":
        return "Okay! What would you like to do today?"
    if kind == "help":
        return (
            "I'm DUM, your daily work-log assistant. I can save your daily updates, answer questions about "
            "your saved logs, create monthly or final PDF reports, edit or delete logs, and manage profiles.\n\n"
            "What would you like to do today?"
        )
    if kind == "talk":
        return f"I'm here{who}. I only handle your daily work log, though. {MENU}"
    return f"I can only help with your daily work log, so I can't help with that one. {MENU}"


def looks_like_profile_statement(text: str) -> bool:
    """True only for explicit statements about the user's own name / role / company."""
    return bool(_PROFILE_GUARD_RE.search(text))


_REPORT_RE = re.compile(
    r"^\s*(please\s+)?(can you\s+|could you\s+)?(generate|create|make|give me|send|prepare|show)\b.*\breport\b"
    r"|\breport\b.*\b(banao|bana do|bana den|chahiye|dikhao)\b"
    r"|^\s*(monthly|weekly)\s+(report|summary)\b",
    re.I,
)
_QUERY_RE = re.compile(
    r"^\s*(what|when|which|who|where|how many|how much|how long|did i|do i|have i|show|list|tell me|"
    r"kya|kab|kon|kaun|kitne|kitna|kis)\b",
    re.I,
)
_DAILY_RE = re.compile(
    r"\b(work(ed|ing)? on|wr(ote|ite)|built|build|fix(ed)?|implement(ed)?|deploy(ed)?|"
    r"complet(ed|e)|finish(ed)?|attend(ed)?|today|aaj|yesterday|kal|check(ed)?[ -]?(in|out)|meeting)\b",
    re.I,
)


_QUESTION_WORD_RE = re.compile(
    r"\b(which|wich|what|when|who|where|how many|how much|kya|kaun|kon|kis|kab|kitne|kitna)\b", re.I
)


def quick_intent(text: str) -> Optional[str]:
    """Cheap keyword routing for clear-cut messages. Returns None -> ask the LLM."""
    t = text.strip()
    if _PROFILE_RE.search(t):
        return "update_profile"
    if _REPORT_RE.search(t):
        return "generate_report"
    if _DELETE_TASK_RE.search(t):
        return "delete_task"
    if _UPDATE_TASK_RE.search(t):
        return "update_task"
    if smalltalk_kind(t):
        return "chat"
    has_cue = bool(_CUE_RE.search(t))
    if (t.endswith("?") or _QUERY_RE.search(t)) and has_cue:
        return "query_history"
    has_question_word = bool(_QUESTION_WORD_RE.search(t))
    has_daily_words = bool(_DAILY_RE.search(t))
    if has_question_word and has_daily_words:
        return None  # ambiguous ("Ahmad work on which project") -> let the LLM decide
    if has_question_word and has_cue:
        return "query_history"
    if has_daily_words and not has_question_word:
        return "daily_update"
    return None  # e.g. "what is AI?" -> the LLM decides (usually chat)


def last_ai_text(messages) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


# --------------------------- friendly error text --------------------------- #
def friendly_error(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    if "429" in low or "ratelimit" in low or "rate limit" in low or "quota" in low:
        return (
            "[rate limit] Azure OpenAI rate/quota limit hit. Wait a minute and try again, "
            "or raise the TPM/RPM quota of your deployment in Azure AI Foundry."
        )
    if "deploymentnotfound" in low or ("404" in low and "deployment" in low):
        return (
            "[config] Deployment not found. AZURE_OPENAI_DEPLOYMENT must be your deployment NAME "
            "(not the model name), and the endpoint must match its resource."
        )
    if "401" in low or "authentication" in low or "access denied" in low:
        return "[auth] Check AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your .env."
    return f"[error] {text}"


# --------------------------- profiles / files --------------------------- #
_SWITCH_RE = re.compile(
    r"^\s*(?:switch to|switch profile to|use profile|open profile|visit profile|go to profile)\s+(.+?)\s*$",
    re.I,
)


def extract_switch_name(text: str) -> Optional[str]:
    """'switch to Wakeel' / "open profile Wakeel's profile" -> 'Wakeel'. None if not a switch command."""
    m = _SWITCH_RE.search(text.strip())
    if not m:
        return None
    name = re.sub(r"(?:'s)?\s+profile$", "", m.group(1).strip(), flags=re.I).strip(" '\"")
    return name or None


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "profile"


# --------------------------- delete / update commands --------------------------- #
ACTIVE_PROFILE = "@active"
_DEL_PROFILE_RES = [
    re.compile(r"^\s*(?:please\s+)?(?:delete|remove)\s+(?:the\s+|this\s+|current\s+|my\s+)?profile\s*$", re.I),
    re.compile(r"^\s*(?:please\s+)?(?:delete|remove)\s+(?:the\s+)?profile\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:please\s+)?(?:delete|remove)\s+(.+?)(?:'s|\s+ki|\s+ka)?\s+profile\s*$", re.I),
    re.compile(r"^\s*(.+?)(?:'s|\s+ki|\s+ka)?\s+profile\s+(?:delete|remove|hata)\b.*$", re.I),
]
_ACTIVE_WORDS = {"this", "current", "my", "active", "this one"}


def extract_delete_profile_name(text: str) -> Optional[str]:
    """'delete profile Wakeel' -> 'Wakeel'; 'delete this profile' -> ACTIVE_PROFILE; else None."""
    for i, rx in enumerate(_DEL_PROFILE_RES):
        m = rx.search(text.strip())
        if m:
            if i == 0:
                return ACTIVE_PROFILE
            name = m.group(1).strip(" '\"")
            return ACTIVE_PROFILE if name.lower() in _ACTIVE_WORDS else (name or None)
    return None


def parse_date(value: Optional[str]) -> Optional[str]:
    """Strict date parser: returns YYYY-MM-DD or None (unlike normalize_date it never defaults to today)."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


YES_WORDS = {"yes", "y", "yeah", "yep", "haan", "han", "ha", "confirm", "confirmed", "ok", "okay", "sure", "delete"}
NO_WORDS = {"no", "n", "nope", "nahi", "nahin", "stop", "cancel"}


def starts_with_word(text: str, words) -> bool:
    parts = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return bool(parts) and parts[0] in words


_DELETE_TASK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:delete|remove)\b.*\b(?:task|tasks|update|updates|log|logs|entry|entries)\b"
    r"|\b(?:task|log|entry)\b.*\b(?:delete|remove|hata)\b.*\b(?:karo|kar do|kardo)\b",
    re.I,
)
_UPDATE_TASK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:update|edit|change|modify|correct)\b.*\b(?:task|tasks|log|logs|entry|entries)\b"
    r"|\b(?:task|log|entry)\b.*\b(?:update|edit|change|badlo|badal)\b.*\b(?:karo|kar do|kardo)\b",
    re.I,
)
