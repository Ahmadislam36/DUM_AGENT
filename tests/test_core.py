import sqlite3
from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from dum_agent import config, database, llm, nodes
from dum_agent.graph import build_graph
from dum_agent.pdf_report import markdown_to_pdf
from dum_agent.schemas import ExtractedLog, IntentResult, ProfileInfo
from dum_agent.utils import (
    compute_stats,
    extract_switch_name,
    friendly_error,
    hours_worked,
    quick_intent,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    database.init_db()


def mk_log(project="Alpha", task="fix bug", d=None):
    return {"date": d or date.today().isoformat(), "project_name": project, "tasks_completed": [task]}


# ---------- basics ----------
def test_hours_and_stats():
    log = {"project_name": "A", "date": "2026-10-01", "time_in": "09:00", "time_out": "5:30 pm"}
    assert hours_worked(log) == 8.5
    assert compute_stats([log])["hours_by_project"] == {"A": 8.5}


# ---------- profiles: separate data per person ----------
def test_profiles_are_isolated():
    a, created = database.upsert_profile("Ahmad", "AI Engineer", "Abc")
    w, _ = database.upsert_profile("Wakeel", "Developer", "XYZ")
    assert created and a["id"] != w["id"]
    database.save_log(mk_log("Alpha"), "raw", a["id"])
    database.save_log(mk_log("Beta"), "raw", w["id"])
    database.save_log(mk_log("Beta2"), "raw", w["id"])
    assert [l["project_name"] for l in database.fetch_30_days_logs(a["id"])] == ["Alpha"]
    assert len(database.fetch_30_days_logs(w["id"])) == 2
    assert len(database.search_logs(profile_id=a["id"], keywords=["beta"])) == 0
    assert len(database.fetch_all_logs(w["id"])) == 2


def test_upsert_is_case_insensitive_and_merges():
    database.upsert_profile("Ahmad", None, None)
    p, created = database.upsert_profile("ahmad", "AI Engineer", "Abc")
    assert not created and len(database.list_profiles()) == 1
    assert (p["role"], p["company"]) == ("AI Engineer", "Abc")
    p, _ = database.upsert_profile("AHMAD", None, "NewCo")
    assert (p["role"], p["company"]) == ("AI Engineer", "NewCo")


def test_migration_from_old_schema(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    conn = sqlite3.connect(old)
    conn.executescript(
        """
        CREATE TABLE daily_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, time_in TEXT, time_out TEXT,
            project_name TEXT, tasks TEXT, meetings TEXT, raw_input TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO daily_logs (date, project_name, tasks) VALUES ('2026-10-01', 'lawgpt', '["CI/CD"]');
        CREATE TABLE user_profile (id INTEGER PRIMARY KEY, name TEXT, role TEXT, company TEXT, updated_at TIMESTAMP);
        INSERT INTO user_profile (id, name, role, company) VALUES (1, 'Ahmad', 'AI Engineer', 'Abc');
        """
    )
    conn.commit(); conn.close()
    monkeypatch.setattr(config, "DB_PATH", old)
    database.init_db()
    database.init_db()  # idempotent
    profiles = database.list_profiles()
    assert [p["name"] for p in profiles] == ["Ahmad"]
    assert len(database.fetch_all_logs(profiles[0]["id"])) == 1


# ---------- routing ----------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("my name is Ahmad, I work as AI engineer at Abc", "update_profile"),
        ("Generate my monthly report", "generate_report"),
        ("generate my final report", "generate_report"),
        ("report banao", "generate_report"),
        ("What did I do last Tuesday?", "query_history"),
        ("show me my work on lawgpt", "query_history"),
        ("Aaj Alpha pe login fix kiya", "daily_update"),
        ("i work on lawgpt and write a CI/CD pipeline", "daily_update"),
        ("I prepared the weekly report for the client today", "daily_update"),
        ("Ahmad work on which project and which company and which role", None),
        ("Ahmad wich company", "query_history"),
        ("hello there", "chat"),
        ("hello i worked on lawgpt today", "daily_update"),
        ("can you talk with me in simple conversation", "chat"),
        ("ok talk with me like me simple conversation", "chat"),
        ("hello", "chat"),
        ("what is AI?", None),
        ("can you tell me about what is AI", None),
        ("show me a joke", None),
        ("i am doing great at digifloat", None),
        ("i work at digifloat", "update_profile"),
    ],
)
def test_quick_intent(text, expected):
    assert quick_intent(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("switch to Wakeel", "Wakeel"),
        ("Switch to Wakeel's profile", "Wakeel"),
        ("open profile Ahmad", "Ahmad"),
        ("use profile Ahmad Khan", "Ahmad Khan"),
        ("I switched to a new laptop", None),
    ],
)
def test_extract_switch_name(text, expected):
    assert extract_switch_name(text) == expected


def test_fast_routing_makes_no_llm_call(monkeypatch):
    def boom(schema):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(nodes, "structured", boom)
    out = nodes.classify_intent_node(
        {"messages": [HumanMessage(content="Aaj Alpha pe login fix kiya")], "profile_id": 1}
    )
    assert out["intent"] == "daily_update"


# ---------- graph flows ----------
def _fake_structured(extract_results):
    class Fake:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, msgs):
            if self.schema is IntentResult:
                return IntentResult(intent="daily_update")
            if self.schema is ExtractedLog:
                return extract_results.pop(0)
            if self.schema is ProfileInfo:
                text = msgs[-1].content
                if "Wakeel" in text:
                    return ProfileInfo(name="Wakeel", role="Developer", company="XYZ")
                if "Ahmad" in text:
                    return ProfileInfo(name="Ahmad", role="AI Engineer", company="Abc")
                return ProfileInfo(role="Lead")  # no name
            return SimpleNamespace(questions=["Which project?"])

    return Fake


def _send(graph, text, pid, thread):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "profile_id": pid},
        {"configurable": {"thread_id": thread}},
    )


def test_needs_profile_guard(monkeypatch):
    monkeypatch.setattr(nodes, "structured", _fake_structured([]))
    out = _send(build_graph(), "Aaj Alpha pe login fix kiya", None, "t0")
    assert "No profile selected" in out["response"]
    assert database.list_profiles() == []


def test_clarify_loop_then_save_scoped(monkeypatch):
    p, _ = database.upsert_profile("Ahmad", "AI Engineer", "Abc")
    results = [
        ExtractedLog(is_clear=False, missing_info_reason="no project"),
        ExtractedLog(project_name="Alpha", tasks_completed=["fixed login"], is_clear=True),
    ]
    monkeypatch.setattr(nodes, "structured", _fake_structured(results))
    graph = build_graph()
    out = _send(graph, "worked a bit today", p["id"], "t1")
    assert "Which project?" in out["response"]
    out = _send(graph, "Alpha, fixed login", p["id"], "t1")
    assert out["response"].startswith("Saved for Ahmad!")
    assert len(database.fetch_30_days_logs(p["id"])) == 1


def test_profile_create_and_switch_flow(monkeypatch):
    monkeypatch.setattr(nodes, "structured", _fake_structured([]))
    graph = build_graph()
    out = _send(graph, "my name is Ahmad, AI engineer at Abc", None, "a")
    assert out["response"].startswith("Created profile") and out["profile_id"]
    ahmad_id = out["profile_id"]

    out = _send(graph, "my name is Wakeel, developer at XYZ", ahmad_id, "a")
    wakeel_id = out["profile_id"]
    assert wakeel_id != ahmad_id and len(database.list_profiles()) == 2

    out = _send(graph, "switch to Ahmad", wakeel_id, "b")
    assert out["profile_id"] == ahmad_id and "Switched to Ahmad" in out["response"]
    out = _send(graph, "switch to Nobody", ahmad_id, "b")
    assert "No profile named" in out["response"] and out["profile_id"] == ahmad_id

    out = _send(graph, "my name is ahmad, company NewCo", ahmad_id, "b")  # same person -> update
    assert out["response"].startswith("Updated profile")
    assert len(database.list_profiles()) == 2


def test_report_is_per_profile_and_creates_pdf(monkeypatch, tmp_path):
    a, _ = database.upsert_profile("Ahmad", "AI Engineer", "Abc")
    w, _ = database.upsert_profile("Wakeel", "Developer", "XYZ")
    database.save_log(mk_log("Alpha"), "raw", a["id"])
    database.save_log(mk_log("Beta", d="2020-01-01"), "raw", w["id"])  # old: only in the final report

    seen = {}

    def fake_llm_text(msgs):
        seen["prompt"] = msgs[-1].content
        return "# Monthly Progress Report\n\n## Summary\n- **Alpha** shipped\n\n| Project | Hours |\n|---|---|\n| Alpha | 8 |\n\nDone \u2014 ok \u2705"

    monkeypatch.setattr(nodes, "llm_text", fake_llm_text)
    graph = build_graph()

    out = _send(graph, "generate my monthly report", a["id"], "r1")
    assert "Alpha" in seen["prompt"] and "Beta" not in seen["prompt"]
    assert out["report_path"].endswith(".pdf")
    assert (tmp_path / "reports" / "ahmad").exists()
    assert open(out["report_path"], "rb").read(4) == b"%PDF"

    out = _send(graph, "generate my monthly report", w["id"], "r2")  # Wakeel: nothing in last 30 days
    assert "no logs" in out["response"] and out["report_path"] == ""

    out = _send(graph, "generate my final report", w["id"], "r3")  # all-time report includes old log
    assert "Beta" in seen["prompt"] and "Report type: Final" in seen["prompt"]
    assert (tmp_path / "reports" / "wakeel").exists()


def test_pdf_handles_markdown_and_unicode(tmp_path):
    md = "# T\n\ntext **bold** *it* `code` \u2014 \u2705 \u0627\n\n- a\n  - b\n1. one\n\n| A | B |\n|---|---|\n| 1 | 2 |\n---\n"
    p = markdown_to_pdf(md, tmp_path / "x.pdf")
    assert p.read_bytes()[:4] == b"%PDF" and p.stat().st_size > 1000


def test_friendly_errors():
    assert "rate limit" in friendly_error(Exception("Error code: 429 - rate limit")).lower()
    assert "Deployment not found" in friendly_error(Exception("DeploymentNotFound"))
    assert "[error]" in friendly_error(ValueError("boom"))


def test_azure_llm_builds(monkeypatch):
    monkeypatch.setattr(config, "AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setattr(config, "AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setattr(config, "AZURE_OPENAI_DEPLOYMENT", "my-deploy")
    llm.get_llm.cache_clear()
    assert type(llm.get_llm()).__name__ == "AzureChatOpenAI"
    monkeypatch.setattr(config, "AZURE_OPENAI_DEPLOYMENT", None)
    llm.get_llm.cache_clear()
    with pytest.raises(RuntimeError):
        llm.get_llm()
    llm.get_llm.cache_clear()
