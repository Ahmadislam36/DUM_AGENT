import os
import uuid

from langchain_core.messages import HumanMessage

from .database import find_profile_by_name, get_profile, init_db, list_profiles
from .graph import build_graph
from .graph_viz import describe_graph
from .logger import get_logger, log_startup
from .runner import run_graph
from .utils import friendly_error


def main() -> None:
    init_db()
    log_startup()
    graph = build_graph()
    log = get_logger("cli")
    base = os.getenv("DUM_THREAD_ID", f"dum-{uuid.uuid4().hex[:8]}")

    profiles = list_profiles()
    current = profiles[0]["id"] if len(profiles) == 1 else None  # auto-select if only one
    last_path = []

    print("=" * 60)
    print(" DUM Agent - Daily Update Management Agent")
    print(" Log updates, ask about history, or say 'generate my monthly report'.")
    print(" Commands: /profiles  /use <name>  /graph  cancel  exit")
    print(" New person: 'My name is Wakeel, I work as developer at XYZ'")
    print("=" * 60)
    if len(profiles) > 1:
        print("Profiles:", ", ".join(p["name"] for p in profiles), "- pick one with /use <name>")

    while True:
        label = (get_profile(current) or {}).get("name", "no profile")
        try:
            user_input = input(f"\nYou [{label}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if low == "/profiles":
            for p in list_profiles():
                print(f"- {p['name']} ({p['role'] or '-'} @ {p['company'] or '-'})")
            continue
        if low == "/graph":
            print(describe_graph(graph, last_path))
            continue
        if low.startswith("/use "):
            p = find_profile_by_name(user_input[5:])
            if p:
                current = p["id"]
                log.info("/use -> switched to profile %r (id=%s)", p["name"], p["id"])
                print(f"Switched to {p['name']}.")
            else:
                print("No such profile. Use /profiles to list them.")
            continue

        try:
            out, trace = run_graph(
                graph,
                {"messages": [HumanMessage(content=user_input)], "profile_id": current},
                {"configurable": {"thread_id": f"{base}-{current}"}},  # separate chat per profile
            )
            last_path = [t["node"] for t in trace]
            print(f"\nDUM Agent: {out.get('response', '(no response)')}")
            current = out.get("profile_id", current)  # profile may have been created/switched
        except Exception as e:  # keep the CLI alive on API/config errors
            print(f"\n{friendly_error(e)}")
