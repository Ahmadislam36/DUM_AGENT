from datetime import date

import pytest
from langchain_core.messages import HumanMessage

from dum_agent import config, database, nodes
from dum_agent.graph import build_graph
from dum_agent.schemas import ExtractedLog, IntentResult, ProfileInfo, DeleteTarget


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    database.init_db()


class Fake:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, msgs):
        if self.schema is ProfileInfo:
            return ProfileInfo(name="Ahmad", role="AI Engineer", company="Abc")
        if self.schema is ExtractedLog:
            return ExtractedLog(project_name="lawgpt", tasks_completed=["CI/CD"], time_in="09:00", time_out="17:00", is_clear=True)
        if self.schema is DeleteTarget:
            return DeleteTarget(task_keyword="CI/CD")
        return IntentResult(intent="daily_update")


def send(g, text, pid):
    return g.invoke({"messages": [HumanMessage(content=text)], "profile_id": pid}, {"configurable": {"thread_id": "t"}})


def test_steps_are_printed_to_terminal(monkeypatch, capsys):
    monkeypatch.setattr(nodes, "structured", Fake)
    monkeypatch.setattr(nodes, "llm_text", lambda m: "# Monthly Progress Report\n\n- ok")
    g = build_graph()

    out = send(g, "my name is Ahmad, AI engineer at Abc", None)
    pid = out["profile_id"]
    send(g, "Aaj lawgpt pe CI/CD pipeline likhi", pid)
    send(g, "generate my monthly report", pid)
    send(g, "delete the CI/CD task from lawgpt", pid)
    send(g, "yes", pid)
    send(g, "delete this profile", pid)
    send(g, "yes", pid)
    text = capsys.readouterr().out

    for expected in [
        "Step 1: message received",
        "intent = update_profile (via keyword shortcut (no LLM call))",
        "Profile created: 'Ahmad'",
        "Extracted: date=",
        "INSERT daily_logs -> log #1",
        "Step 5: done -> log #1 saved",
        "Monthly report requested",
        "PDF created",
        "match(es) in Ahmad's logs",
        "confirmed -> deleting",
        "DELETE profile id=",
    ]:
        assert expected in text, f"missing in terminal output: {expected}"
