from datetime import date

import pytest
from langchain_core.messages import HumanMessage

from dum_agent import config, database, nodes
from dum_agent.graph import build_graph
from dum_agent.schemas import DeleteTarget, EditTarget
from dum_agent.utils import ACTIVE_PROFILE, extract_delete_profile_name, quick_intent


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    database.init_db()


def add_log(pid, project, tasks, d=None):
    return database.save_log(
        {"date": d or date.today().isoformat(), "project_name": project, "tasks_completed": tasks}, "raw", pid
    )


def fake(target):
    class F:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, msgs):
            assert self.schema in (DeleteTarget, EditTarget), "no other LLM call expected"
            return target

    return F


def send(graph, text, pid, thread="t"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "profile_id": pid},
        {"configurable": {"thread_id": thread}},
    )


# ---------- command detection ----------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("delete profile Wakeel", "Wakeel"),
        ("Delete Wakeel's profile", "Wakeel"),
        ("wakeel ki profile delete karo", "wakeel"),
        ("delete this profile", ACTIVE_PROFILE),
        ("delete profile", ACTIVE_PROFILE),
        ("Aaj profile page pe kaam kiya", None),
    ],
)
def test_extract_delete_profile_name(text, expected):
    assert extract_delete_profile_name(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("delete the CI/CD task from lawgpt", "delete_task"),
        ("remove log #5", "delete_task"),
        ("task delete karo", "delete_task"),
        ("update the login task to: fixed login and signup", "update_task"),
        ("edit log 3 project to Alpha", "update_task"),
        ("Updated docs today", "daily_update"),
        ("I worked on update task API endpoint today", "daily_update"),
    ],
)
def test_quick_intent_delete_update(text, expected):
    assert quick_intent(text) == expected


# ---------- database ops ----------
def test_db_remove_task_and_scoping():
    a, _ = database.upsert_profile("Ahmad")
    w, _ = database.upsert_profile("Wakeel")
    log = add_log(a["id"], "lawgpt", ["CI/CD", "docs"])
    assert database.remove_task(w["id"], log, 0) == "not_found"  # other profile cannot touch it
    assert database.remove_task(a["id"], log, 0, expected_text="WRONG") == "not_found"
    assert database.remove_task(a["id"], log, 0, expected_text="CI/CD") == "task_deleted"
    assert database.get_log(a["id"], log)["tasks"] == ["docs"]
    assert database.remove_task(a["id"], log, 0) == "log_deleted"
    assert database.get_log(a["id"], log) is None


def test_db_update_log():
    a, _ = database.upsert_profile("Ahmad")
    log = add_log(a["id"], "lawgpt", ["CI/CD", "docs"])
    u = database.update_log(a["id"], log, task_index=1, new_task_text="API docs", project_name="LawGPT v2")
    assert u["tasks"] == ["CI/CD", "API docs"] and u["project_name"] == "LawGPT v2"


# ---------- delete profile ----------
def test_delete_profile_needs_confirmation_and_keeps_others():
    a, _ = database.upsert_profile("Ahmad")
    w, _ = database.upsert_profile("Wakeel")
    add_log(a["id"], "A", ["x"])
    add_log(w["id"], "W", ["y"])
    add_log(w["id"], "W2", ["z"])
    g = build_graph()

    out = send(g, "delete profile Wakeel", a["id"])
    assert "2 log(s)" in out["response"] and database.find_profile_by_name("Wakeel")  # nothing deleted yet

    out = send(g, "maybe", a["id"])
    assert "reply 'yes'" in out["response"].lower() and database.find_profile_by_name("Wakeel")

    out = send(g, "yes", a["id"])
    assert "Deleted profile 'Wakeel' and 2 log(s)" in out["response"]
    assert database.find_profile_by_name("Wakeel") is None and out["profile_id"] == a["id"]
    assert len(database.fetch_all_logs(a["id"])) == 1  # Ahmad untouched
    assert out["deleted_profile_id"] == w["id"]


def test_delete_active_profile_falls_back_and_cancel_works():
    a, _ = database.upsert_profile("Ahmad")
    w, _ = database.upsert_profile("Wakeel")
    g = build_graph()
    send(g, "delete this profile", w["id"], "x")
    out = send(g, "cancel", w["id"], "x")
    assert "cancelled" in out["response"].lower() and database.find_profile_by_name("Wakeel")

    send(g, "delete this profile", w["id"], "x")
    out = send(g, "yes", w["id"], "x")
    assert out["profile_id"] == a["id"]  # fell back to the remaining profile


# ---------- delete task ----------
def test_delete_single_task_with_confirm(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    log = add_log(a["id"], "lawgpt", ["CI/CD pipeline", "write docs"])
    monkeypatch.setattr(nodes, "structured", fake(DeleteTarget(task_keyword="CI/CD", project_name="lawgpt")))
    g = build_graph()
    out = send(g, "delete the CI/CD task from lawgpt", a["id"])
    assert "Delete this task" in out["response"]
    assert database.get_log(a["id"], log)["tasks"] == ["CI/CD pipeline", "write docs"]
    out = send(g, "yes", a["id"])
    assert out["response"].startswith("Deleted")
    assert database.get_log(a["id"], log)["tasks"] == ["write docs"]


def test_delete_pick_from_many_and_profile_scope(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    w, _ = database.upsert_profile("Wakeel")
    l1 = add_log(a["id"], "lawgpt", ["fix bug A"], d="2026-10-01")
    l2 = add_log(a["id"], "lawgpt", ["fix bug B"], d="2026-10-02")
    lw = add_log(w["id"], "lawgpt", ["fix bug W"], d="2026-10-02")
    monkeypatch.setattr(nodes, "structured", fake(DeleteTarget(task_keyword="fix bug")))
    g = build_graph()
    out = send(g, "delete the fix bug task", a["id"])
    assert "2 matches" in out["response"] and "bug W" not in out["response"]  # only Ahmad's tasks listed
    out = send(g, "1", a["id"])  # newest first -> 1 = bug B (log l2)
    assert out["response"].startswith("Deleted") and database.get_log(a["id"], l2) is None
    assert database.get_log(a["id"], l1) and database.get_log(w["id"], lw)


def test_delete_whole_log_and_not_found(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    log = add_log(a["id"], "lawgpt", ["t1", "t2"])
    g = build_graph()
    monkeypatch.setattr(nodes, "structured", fake(DeleteTarget(log_id=log, delete_whole_log=True)))
    out = send(g, "remove log #%d" % log, a["id"])
    assert "whole update" in out["response"]
    send(g, "yes", a["id"])
    assert database.get_log(a["id"], log) is None

    monkeypatch.setattr(nodes, "structured", fake(DeleteTarget(task_keyword="nothing-here")))
    out = send(g, "delete the nothing-here task", a["id"], "t2")
    assert "couldn't find" in out["response"]


# ---------- update task ----------
def test_update_single_task_immediately(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    log = add_log(a["id"], "lawgpt", ["CI/CD pipeline", "docs"])
    monkeypatch.setattr(
        nodes, "structured",
        fake(EditTarget(task_keyword="CI/CD", new_task_text="CI/CD with GitHub Actions")),
    )
    out = send(build_graph(), "update the CI/CD task to: CI/CD with GitHub Actions", a["id"])
    assert "Updated log" in out["response"]
    assert database.get_log(a["id"], log)["tasks"] == ["CI/CD with GitHub Actions", "docs"]


def test_update_log_level_field_with_pick(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    l1 = add_log(a["id"], "lawgpt", ["x"], d="2026-10-01")
    l2 = add_log(a["id"], "lawgpt", ["y"], d="2026-10-02")
    monkeypatch.setattr(nodes, "structured", fake(EditTarget(project_name="lawgpt", new_project_name="LawGPT")))
    g = build_graph()
    out = send(g, "change project name of lawgpt entries to LawGPT", a["id"])
    assert "2 matches" in out["response"]
    send(g, "1,2", a["id"])
    assert database.get_log(a["id"], l1)["project_name"] == "LawGPT"
    assert database.get_log(a["id"], l2)["project_name"] == "LawGPT"


def test_update_without_changes_asks(monkeypatch):
    a, _ = database.upsert_profile("Ahmad")
    monkeypatch.setattr(nodes, "structured", fake(EditTarget(task_keyword="x")))
    out = send(build_graph(), "edit the x task", a["id"])
    assert "What should I change" in out["response"]


def test_needs_profile_for_delete():
    out = send(build_graph(), "delete the CI/CD task", None)
    assert "No profile selected" in out["response"]
