"""Draw / describe the compiled LangGraph (DOT for Streamlit, plain text for the CLI)."""
from collections import defaultdict
from typing import List, Optional

_NAMES = {"__start__": "START", "__end__": "END"}


_THEMES = {
    "light": dict(fill="#f1f3f5", line="#868e96", font="#212529", edge="#adb5bd",
                  hot_fill="#ffd43b", hot_line="#e8590c", hot_font="#212529", hot_edge="#e8590c"),
    "dark": dict(fill="#15213A", line="#25365A", font="#E8EDF6", edge="#4A5C80",
                 hot_fill="#F5C04A", hot_line="#F5C04A", hot_font="#0E1726", hot_edge="#F5C04A"),
}


def graph_to_dot(graph, path: Optional[List[str]] = None, theme: str = "light") -> str:
    """Graphviz DOT. Nodes/edges of the last executed `path` are highlighted."""
    t = _THEMES.get(theme, _THEMES["light"])
    g = graph.get_graph()
    path = path or []
    seq = ["__start__"] + path + (["__end__"] if path else [])
    hot_nodes, hot_edges = set(seq), set(zip(seq, seq[1:]))

    lines = [
        "digraph G {",
        "rankdir=TB;",
        'bgcolor="transparent";',
        f'node [shape=box, style="rounded,filled", fillcolor="{t["fill"]}", color="{t["line"]}", fontcolor="{t["font"]}", fontname="Helvetica"];',
        f'edge [color="{t["edge"]}"];',
    ]
    for nid in g.nodes:
        attrs = [f'label="{_NAMES.get(nid, nid)}"']
        if nid in _NAMES:
            attrs.append("shape=ellipse")
        if path and nid in hot_nodes:
            attrs.append(f'fillcolor="{t["hot_fill"]}", color="{t["hot_line"]}", fontcolor="{t["hot_font"]}", penwidth=2')
        lines.append(f'"{nid}" [{", ".join(attrs)}];')
    for e in g.edges:
        attrs = ['style="dashed"'] if e.conditional else []
        if (e.source, e.target) in hot_edges:
            attrs += [f'color="{t["hot_edge"]}"', "penwidth=2.5"]
        lines.append(f'"{e.source}" -> "{e.target}" [{", ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)


def describe_graph(graph, path: Optional[List[str]] = None) -> str:
    """Plain-text map: every node and where it can go next ('?' = chosen at run time)."""
    g = graph.get_graph()
    out = defaultdict(list)
    for e in g.edges:
        out[e.source].append(_NAMES.get(e.target, e.target) + ("?" if e.conditional else ""))
    lines = ["Graph map (? = route chosen at run time):"]
    for nid in g.nodes:
        if nid == "__end__":
            continue
        lines.append(f"  {_NAMES.get(nid, nid):<16} -> {', '.join(out[nid])}")
    if path:
        lines.append("Last run: " + " -> ".join(["START"] + path + ["END"]))
    return "\n".join(lines)
