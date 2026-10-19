import pytest
from langchain_core.messages import HumanMessage

from dum_agent import config, database, nodes
from dum_agent.graph import build_graph
from dum_agent.graph_viz import describe_graph, graph_to_dot
from dum_agent.runner import run_graph
from dum_agent.schemas import ExtractedLog, IntentResult, ProfileInfo, SearchFilter


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
            return ProfileInfo(name="Ahmad")
        if self.schema is ExtractedLog:
            return ExtractedLog(project_name="lawgpt", tasks_completed=["CI/CD"], is_clear=True)
        if self.schema is SearchFilter:
            return SearchFilter()
        return IntentResult(intent="daily_update")


def go(graph, text, pid):
    return run_graph(graph, {"messages": [HumanMessage(content=text)], "profile_id": pid}, {"configurable": {"thread_id": "t"}})


def test_paths_and_terminal_trace(monkeypatch, capsys):
    monkeypatch.setattr(nodes, "structured", Fake)
    monkeypatch.setattr(nodes, "llm_text", lambda m: "# Monthly Progress Report\n\n- ok")
    g = build_graph()

    out, trace = go(g, "my name is Ahmad", None)
    assert [t["node"] for t in trace] == ["classify_intent", "update_profile"]
    pid = out["profile_id"]

    out, trace = go(g, "Aaj lawgpt pe CI/CD pipeline likhi", pid)
    assert [t["node"] for t in trace] == ["classify_intent", "extract_data", "save_to_db"]
    assert out["response"].startswith("Saved for Ahmad")

    _, trace = go(g, "generate my monthly report", pid)
    assert [t["node"] for t in trace] == ["classify_intent", "fetch_data", "generate_report"]

    _, trace = go(g, "What did I do today?", pid)
    assert [t["node"] for t in trace][-2:] == ["search_history", "format_response"]

    _, trace = go(g, "Aaj lawgpt pe kaam kiya", None)  # no profile -> stops right after the classifier
    assert [t["node"] for t in trace] == ["classify_intent"]

    text = capsys.readouterr().out
    assert "PATH: START -> classify_intent -> extract_data -> save_to_db -> END" in text
    assert "Node 2: extract_data" in text and "is_clear=True" in text


def test_graph_drawing(monkeypatch):
    g = build_graph()
    dot = graph_to_dot(g, ["classify_intent", "extract_data", "save_to_db"])
    assert dot.startswith("digraph G {") and '"classify_intent" -> "extract_data"' in dot
    assert dot.count("#e8590c") >= 6  # highlighted nodes + edges
    assert "extract_data" in describe_graph(g, ["classify_intent"]) and "Last run" in describe_graph(g, ["classify_intent"])
