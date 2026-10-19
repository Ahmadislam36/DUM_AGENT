"""Runs the LangGraph node by node and prints the path it takes (terminal trace)."""
import time
from typing import Any, Dict, List, Tuple

from .logger import get_logger

log = get_logger("graph")
_SHOW_VALUE = ("intent", "is_clear", "profile_id", "report_kind", "awaiting_clarification")


def _summarize(update: Any) -> str:
    if not isinstance(update, dict):
        return ""
    parts = []
    for key, value in update.items():
        if key == "messages":
            continue
        if key in _SHOW_VALUE:
            parts.append(f"{key}={value}")
        elif key == "pending_action":
            parts.append("pending_action=" + (value.get("type", "set") if value else "cleared"))
        elif key == "logs":
            parts.append(f"logs={len(value)} row(s)")
        else:
            parts.append(key)
    return ", ".join(parts)


def run_graph(graph, payload: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one turn. Returns (final_state, trace) where trace = [{'node', 'seconds', 'summary'}, ...]."""
    trace: List[Dict[str, Any]] = []
    log.info("=== Graph run started ===")
    last = time.perf_counter()
    try:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                now = time.perf_counter()
                step = {"node": node, "seconds": round(now - last, 2), "summary": _summarize(update)}
                last = now
                trace.append(step)
                log.info(
                    "Node %d: %-16s done (%.1fs) | updated: %s",
                    len(trace), node, step["seconds"], step["summary"] or "nothing",
                )
    except Exception as e:
        path = " -> ".join(["START"] + [t["node"] for t in trace])
        log.error("Graph stopped: %s -> (error in the next step) %s: %s", path, type(e).__name__, str(e)[:150])
        raise
    log.info("PATH: %s", " -> ".join(["START"] + [t["node"] for t in trace] + ["END"]))
    return graph.get_state(config).values, trace
