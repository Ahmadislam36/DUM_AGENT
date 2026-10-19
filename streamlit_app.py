import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

from dum_agent import config
from dum_agent.database import count_logs, delete_profile, init_db, list_profiles, search_logs, upsert_profile
from dum_agent.graph import build_graph
from dum_agent.graph_viz import graph_to_dot
from dum_agent.logger import get_logger, log_startup
from dum_agent.runner import run_graph
from dum_agent.ui_theme import CSS, build_color_map, entries_html, hero_html, initials, ribbon_card, stat_strip
from dum_agent.utils import compute_stats, friendly_error, slugify

st.set_page_config(page_title="DUM - Daily work log", page_icon="🗒️", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
init_db()
ui = get_logger("ui")
ss = st.session_state

if "graph" not in ss:
    ss.graph = build_graph()  # MemorySaver lives per browser session
    ss.sid = uuid.uuid4().hex[:8]
    ss.histories = {}  # chat history per profile
    ss.nonce = {}  # "new chat" counter per profile
    log_startup()
    ui.info("New browser session %s started", ss.sid)

EXAMPLE_UPDATE = "Today I worked on LawGPT and wrote the CI/CD pipeline, 9 to 5"
EXAMPLE_PROFILE = "My name is Ahmad, I work as AI engineer at Abc"


def thread_id(pid) -> str:
    return f"{ss.sid}-{pid}-{ss.nonce.get(pid, 0)}"  # separate agent memory per profile


def ask(text: str, pid):
    try:
        out, trace = run_graph(
            ss.graph,
            {"messages": [HumanMessage(content=text)], "profile_id": pid},
            {"configurable": {"thread_id": thread_id(pid)}},
        )
        ss.last_trace = trace  # shown in the Graph tab
        ss.last_prompt = text
    except Exception as e:
        return friendly_error(e), {}
    return out.get("response") or f"(no response) intent={out.get('intent')!r}", out


def profile_form(key: str):
    """Create a profile, or update it when the name already exists."""
    with st.form(key, clear_on_submit=True):
        n = st.text_input("Name")
        r = st.text_input("Role", placeholder="AI engineer")
        c = st.text_input("Company", placeholder="Abc")
        if st.form_submit_button("Save profile", type="primary"):
            if n.strip():
                ui.info("Profile form submitted for %r", n.strip())
                prof, _ = upsert_profile(n.strip(), r.strip() or None, c.strip() or None)
                ss["switch_to"] = prof["id"]
                st.rerun()
            else:
                st.warning("Add a name to save the profile.")


# ---------------- profiles + sidebar ----------------
profiles = list_profiles()
by_id = {p["id"]: p for p in profiles}
by_name = {p["name"]: p for p in profiles}  # names are unique (case-insensitive)

if "switch_to" in ss:  # must be applied before the selectbox is created
    target = by_id.get(ss.pop("switch_to"))
    if target:
        ss["profile_select"] = target["name"]

quick = ss.pop("chip_prompt", None)  # set by the suggestion buttons below

with st.sidebar:
    st.markdown('<div class="brand">DUM</div><div class="brand-tag">Daily work log</div>', unsafe_allow_html=True)

    if by_name:
        if ss.get("profile_select") not in by_name:
            ss["profile_select"] = next(iter(by_name))
        name = st.selectbox("Active profile", list(by_name), key="profile_select")
        cur = by_name[name]
        pid = cur["id"]
    else:
        pid, cur = None, None

    with st.expander("Add or edit a profile", expanded=not by_name):
        st.caption("A new name creates a profile. An existing name updates it.")
        profile_form("profile_form_side")

    if pid is not None:
        st.markdown("<div class='hint' style='margin-top:1rem'>Shortcuts</div>", unsafe_allow_html=True)
        if st.button("Monthly report", key="act_month"):
            quick = "generate my monthly report"
        if st.button("Final report (all time)", key="act_final"):
            quick = "generate my final report"
        if st.button("What did I do this week?", key="act_week"):
            quick = "What did I do this week?"
        if st.button("Start a new chat", key="act_new"):
            ss.nonce[pid] = ss.nonce.get(pid, 0) + 1
            ss.histories[pid] = []
            st.rerun()

        with st.expander("Delete this profile"):
            st.warning(f"This deletes {cur['name']} and all {count_logs(pid)} of its updates. It cannot be undone.")
            sure = st.checkbox("Yes, delete it", key=f"sure_{pid}")
            if st.button("Delete profile", key="act_delete", disabled=not sure):
                ui.info("Delete profile button confirmed for %r", cur["name"])
                delete_profile(pid)
                ss.histories.pop(pid, None)
                ss.pop("profile_select", None)
                st.rerun()

# chat_input is pinned to the bottom of the page wherever it is called
prompt = st.chat_input("Tell me what you worked on, or ask about your logs...") or quick

# ---------------- header ----------------
if pid is not None:
    st.markdown(hero_html(cur), unsafe_allow_html=True)
    if len(profiles) > 1:  # one-click profile switcher
        shown = profiles[:6]
        cols = st.columns([1] * len(shown) + [max(1, 8 - len(shown))])
        for col, p in zip(cols, shown):
            if col.button(f"{initials(p['name'])}  {p['name']}", key=f"person_{p['id']}",
                          type="primary" if p["id"] == pid else "secondary"):
                if p["id"] != pid:
                    ss["switch_to"] = p["id"]
                    st.rerun()

tab_chat, tab_log, tab_graph, tab_reports = st.tabs(["Chat", "Logbook", "Graph", "Reports"])

# ---------------- chat ----------------
with tab_chat:
    if pid is None:
        st.markdown(
            '<div class="welcome"><h3>Start with a profile</h3>'
            "<p>Each person gets their own updates, history and reports.</p>"
            f"<p>Type <code>{EXAMPLE_PROFILE}</code> below, or fill in the form.</p></div>",
            unsafe_allow_html=True,
        )
        profile_form("profile_form_main")

    ribbon_slot = st.empty()
    hist = ss.histories.setdefault(pid, [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    out = {}
    if prompt:
        ui.info("Chat message received (profile_id=%s): %r", pid, prompt[:80])
        hist.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Working on it..."):
                reply, out = ask(prompt, pid)
            st.markdown(reply)
            rp = out.get("report_path")
            if out.get("intent") == "generate_report" and rp and Path(rp).exists():
                st.download_button(
                    "Download PDF", Path(rp).read_bytes(), file_name=Path(rp).name,
                    mime="application/pdf", key=f"dl-{uuid.uuid4().hex}",
                )
        hist.append({"role": "assistant", "content": reply})

    if pid is not None:  # week ribbon, drawn after the message so it is always up to date
        week_logs = search_logs(profile_id=pid, limit=500)
        ribbon_slot.markdown(
            ribbon_card(week_logs, build_color_map(week_logs), title="This week"), unsafe_allow_html=True
        )

    if pid is not None and not hist and not prompt:
        st.markdown(
            '<div class="welcome"><h3>What did you work on?</h3>'
            f"<p>Write it the way you would tell a colleague, for example <code>{EXAMPLE_UPDATE}</code>.</p>"
            "<p>I will ask a short follow-up if the project or tasks are unclear. I only handle your work log: updates, questions about your logs, reports and profiles.</p></div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("What did I do this week?", key="chip_week"):
            ss["chip_prompt"] = "What did I do this week?"
            st.rerun()
        if c2.button("Generate my monthly report", key="chip_month"):
            ss["chip_prompt"] = "generate my monthly report"
            st.rerun()
        if c3.button("Show my role and company", key="chip_role"):
            ss["chip_prompt"] = "What is my role and company?"
            st.rerun()

    # a profile was created / switched / deleted through chat -> refresh the sidebar and header
    new_pid = out.get("profile_id")
    if out.get("intent") == "apply_pending" and out.get("deleted_profile_id"):
        ss.histories.pop(out["deleted_profile_id"], None)
        if new_pid:
            ss.histories.setdefault(new_pid, []).extend(hist[-2:])  # keep the confirmation visible
            ss["switch_to"] = new_pid
        else:
            ss.pop("profile_select", None)
        st.rerun()
    elif new_pid and new_pid != pid:
        ss.histories.setdefault(new_pid, []).extend(hist[-2:])  # keep the confirmation visible
        ss["switch_to"] = new_pid
        st.rerun()

# ---------------- logbook ----------------
with tab_log:
    logs = search_logs(profile_id=pid, limit=500) if pid is not None else []
    if not logs:
        st.markdown(
            '<div class="welcome"><h3>Nothing logged yet</h3>'
            "<p>Your updates will appear here, grouped by day. Use the Chat tab to add the first one.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        cmap = build_color_map(logs)
        choice = st.selectbox("Project", ["All projects"] + sorted(cmap, key=str.lower), key="logbook_project")
        view = logs if choice == "All projects" else [l for l in logs if l["project_name"] == choice]
        stats = compute_stats(view)
        st.markdown(
            stat_strip([
                ("Updates", stats["entries"]),
                ("Days logged", stats["days_logged"]),
                ("Hours tracked", stats["total_hours"]),
                ("Projects", len({l["project_name"] for l in view})),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown(ribbon_card(view, cmap, title="Last 14 days", days=14, height=170), unsafe_allow_html=True)
        st.markdown(entries_html(view, cmap), unsafe_allow_html=True)

        with st.expander("Table view"):
            df = pd.DataFrame(
                [
                    {"log #": l["id"], "date": l["date"], "project": l["project_name"],
                     "tasks": "; ".join(l["tasks"]), "time in": l.get("time_in"),
                     "time out": l.get("time_out"), "meetings": l.get("meetings")}
                    for l in reversed(view)
                ]
            )
            st.dataframe(df, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{slugify(cur['name'])}-logs.csv", mime="text/csv")

# ---------------- graph ----------------
with tab_graph:
    trace = ss.get("last_trace", [])
    path = [t["node"] for t in trace]
    st.markdown(
        "<div class='hint'>The path your last message took through the agent. Highlighted steps ran.</div>",
        unsafe_allow_html=True,
    )
    st.graphviz_chart(graph_to_dot(ss.graph, path, theme="dark"))
    if trace:
        st.markdown(
            f"<div class='hint'>Last message: <code>{ss.get('last_prompt', '')[:80].replace('<', '&lt;')}</code></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(trace), hide_index=True)
    else:
        st.info("Send a message in the Chat tab to see its path here.")

# ---------------- reports ----------------
with tab_reports:
    rdir = config.REPORTS_DIR / slugify(cur["name"]) if pid is not None else None
    stems = sorted({f.stem for f in rdir.glob("*.*")}, reverse=True) if rdir and rdir.exists() else []
    if not stems:
        st.markdown(
            '<div class="welcome"><h3>No reports yet</h3>'
            "<p>Ask for a monthly or final report in the Chat tab, or use the shortcuts in the sidebar. "
            "Each report is saved as a PDF you can download here.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        pick = st.selectbox("Report", stems)
        pdf, md = rdir / f"{pick}.pdf", rdir / f"{pick}.md"
        d1, d2, _ = st.columns([1, 1.4, 3])
        if pdf.exists():
            d1.download_button("Download PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
        if md.exists():
            d2.download_button("Download Markdown", md.read_text(encoding="utf-8"), file_name=md.name)
            with st.container(key="paper"):
                st.markdown(md.read_text(encoding="utf-8"))
