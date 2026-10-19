"""Regression tests for the behaviour seen in the screenshots (chat treated as work updates,
stuck in follow-up questions, casual sentences changing the profile)."""
import pytest
from langchain_core.messages import HumanMessage

from dum_agent import config, database, nodes
from dum_agent.graph import build_graph
from dum_agent.runner import run_graph
from dum_agent.schemas import ExtractedLog, IntentResult, ProfileInfo, SearchFilter


LLM_TEXT_CALLS = []


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    database.init_db()
    LLM_TEXT_CALLS.clear()
    monkeypatch.setattr(nodes, "llm_text", lambda m: (LLM_TEXT_CALLS.append(1), "REPORT-TEXT")[1])


def fake_llm(monkeypatch, classify=None, extract=None, questions=("Which project?",), profile=None):
    """classify(text) -> intent; extract: list of ExtractedLog returned in order."""
    extract = list(extract or [])
    seen = {"classifier_prompts": [], "profile_calls": 0}

    class Q:
        pass

    class Fake:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, msgs):
            if self.schema is IntentResult:
                text = msgs[-1].content
                seen["classifier_prompts"].append(text)
                return IntentResult(intent=(classify or (lambda t: "daily_update"))(text))
            if self.schema is ExtractedLog:
                return extract.pop(0)
            if self.schema is ProfileInfo:
                seen["profile_calls"] += 1
                return profile or ProfileInfo()
            if self.schema is SearchFilter:
                return SearchFilter()
            q = Q(); q.questions = list(questions); return q

    monkeypatch.setattr(nodes, "structured", Fake)
    return seen


def go(graph, text, pid, thread="t"):
    out, trace = run_graph(graph, {"messages": [HumanMessage(content=text)], "profile_id": pid},
                           {"configurable": {"thread_id": thread}})
    return out, [t["node"] for t in trace]


def unclear():
    return ExtractedLog(is_clear=False, missing_info_reason="no project")


def clear():
    return ExtractedLog(project_name="Alpha", tasks_completed=["fixed login"], is_clear=True)


def snapshot(pid):
    return (database.fetch_all_logs(pid), database.get_profile(pid))


# ---------- 1. greetings get a greeting back; everything off-topic is declined (no LLM answer) ----------
def test_greeting_gets_only_a_greeting_then_asks_for_work(monkeypatch):
    fake_llm(monkeypatch)
    p, _ = database.upsert_profile("Wakeel", "AI engineer", "digifloat")
    before = snapshot(p["id"])
    g = build_graph()
    out, path = go(g, "hello", p["id"])
    assert path == ["classify_intent", "chat_reply"]
    assert out["response"].startswith("Hello Wakeel!") and "What would you like to do today?" in out["response"]
    out, _ = go(g, "Assalam o alaikum", p["id"])
    assert out["response"].startswith("Walaikum assalam Wakeel!")
    out, _ = go(g, "thanks", p["id"])
    assert "Anything else for your work log" in out["response"]
    out, _ = go(g, "can you talk with me in simple conversation", p["id"])
    assert "only handle your daily work log" in out["response"] and "What would you like to do today?" in out["response"]
    assert LLM_TEXT_CALLS == [] and snapshot(p["id"]) == before and not out.get("awaiting_clarification")


def test_off_topic_questions_are_declined_not_answered(monkeypatch):
    fake_llm(monkeypatch, classify=lambda t: "chat")  # the classifier calls them chat
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    for text in ["can you tell me about what is AI", "tell me a joke", "why you update the profile"]:
        out, path = go(g, text, p["id"])
        assert path == ["classify_intent", "chat_reply"]
        assert out["response"].startswith("I can only help with your daily work log")
        assert "What would you like to do today?" in out["response"]
    assert LLM_TEXT_CALLS == []  # no LLM answer was ever written


def test_chat_works_without_any_profile(monkeypatch):
    fake_llm(monkeypatch)
    out, path = go(build_graph(), "hello", None)
    assert path == ["classify_intent", "chat_reply"] and out["response"].startswith("Hello! ")
    assert database.list_profiles() == []


def test_help_explains_what_the_agent_can_do(monkeypatch):
    fake_llm(monkeypatch)
    out, _ = go(build_graph(), "what can you do", None)
    assert "daily work-log assistant" in out["response"] and "PDF reports" in out["response"]


def test_llm_failure_defaults_to_a_polite_decline_not_a_work_update(monkeypatch):
    class Broken:
        def __init__(self, schema): pass
        def invoke(self, msgs): return None

    monkeypatch.setattr(nodes, "structured", Broken)
    p, _ = database.upsert_profile("Wakeel")
    out, path = go(build_graph(), "something unclear", p["id"])
    assert path == ["classify_intent", "chat_reply"] and "only help with your daily work log" in out["response"]


# ---------- 2. never trapped in follow-up questions ----------
def test_report_request_leaves_the_clarification(monkeypatch):
    fake_llm(monkeypatch, extract=[unclear()])
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    out, path = go(g, "worked a bit today", p["id"])
    assert path[-1] == "clarify" and out["awaiting_clarification"]
    out, path = go(g, "generate my monthly report", p["id"])
    assert path[:2] == ["classify_intent", "fetch_data"] and not out["awaiting_clarification"]


def test_general_question_while_waiting_leaves_the_clarification(monkeypatch):
    seen = fake_llm(monkeypatch, classify=lambda t: "chat" if "what is AI" in t else "daily_update", extract=[unclear()])
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    go(g, "worked a bit today", p["id"])
    out, path = go(g, "can you tell me about what is AI", p["id"])
    assert path == ["classify_intent", "chat_reply"] and not out["awaiting_clarification"]
    assert out["response"].startswith("I can only help with your daily work log")
    assert "just asked" in seen["classifier_prompts"][-1]  # the classifier was told what the assistant asked


def test_real_answers_still_continue_the_draft(monkeypatch):
    fake_llm(monkeypatch, extract=[unclear(), clear()])
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    go(g, "worked a bit today", p["id"])
    out, path = go(g, "Alpha, fixed login", p["id"])
    assert path == ["classify_intent", "extract_data", "save_to_db"] and out["response"].startswith("Saved for Wakeel")
    assert out["clarify_rounds"] == 0


def test_agent_gives_up_after_two_follow_up_rounds(monkeypatch):
    fake_llm(monkeypatch, extract=[unclear(), unclear(), unclear()])
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    out, _ = go(g, "worked a bit today", p["id"]);  assert out["clarify_rounds"] == 1
    out, _ = go(g, "dunno", p["id"]);               assert out["clarify_rounds"] == 2
    out, path = go(g, "still dunno", p["id"])
    assert path[-1] == "clarify" and "did not save anything" in out["response"]
    assert not out["awaiting_clarification"] and database.fetch_all_logs(p["id"]) == []


def test_cancel_still_works(monkeypatch):
    fake_llm(monkeypatch, extract=[unclear()])
    p, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    go(g, "worked a bit today", p["id"])
    out, _ = go(g, "cancel", p["id"])
    assert "discarded" in out["response"] and not out["awaiting_clarification"]


# ---------- 3. casual sentences never change the profile ----------
def test_casual_sentence_does_not_update_profile_even_if_llm_picks_update_profile(monkeypatch):
    seen = fake_llm(monkeypatch, classify=lambda t: "update_profile", profile=ProfileInfo(company="digifloat"))
    p, _ = database.upsert_profile("Wakeel", "AI engineer", "Abc")
    out, path = go(build_graph(), "i am doing great at digifloat", p["id"])
    assert path == ["classify_intent", "update_profile"] and "normal message" in out["response"]
    assert database.get_profile(p["id"])["company"] == "Abc" and seen["profile_calls"] == 0


def test_explicit_company_statement_does_update_the_active_profile(monkeypatch):
    fake_llm(monkeypatch, profile=ProfileInfo(company="digifloat"))
    p, _ = database.upsert_profile("Wakeel", "AI engineer", "Abc")
    out, path = go(build_graph(), "I work at digifloat", p["id"])
    assert out["response"].startswith("Updated profile") and database.get_profile(p["id"])["company"] == "digifloat"


# ---------- 4. questions about the user's own logs still work ----------
def test_questions_about_my_logs_still_use_the_log_search(monkeypatch):
    fake_llm(monkeypatch)
    p, _ = database.upsert_profile("Wakeel")
    out, path = go(build_graph(), "What did I do last Tuesday?", p["id"])
    assert path == ["classify_intent", "search_history", "format_response"]
