"""Look and feel of the Streamlit app: CSS + small HTML builders (all user text is escaped)."""
import html
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from .utils import hours_worked, parse_time_minutes

# Project colours (assigned in alphabetical order so they stay stable between views)
PALETTE = ["#4FD1C5", "#F5C04A", "#B794F4", "#FF8A80", "#63B3ED", "#A3D977"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{--night:#0E1726;--deck:#15213A;--deck2:#1B2A47;--hull:#25365A;--chalk:#E8EDF6;--fog:#8E9DBA;
--tide:#4FD1C5;--gold:#F5C04A;--paper:#F4F6FA;--ink:#17233A;--r-lg:18px;--r-md:12px;--r-sm:8px;
--display:'Bricolage Grotesque','Instrument Sans',system-ui,sans-serif}
.stApp{background:var(--night);color:var(--chalk);font-family:'Instrument Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp h1,.stApp h2,.stApp h3{font-family:var(--display);letter-spacing:-.015em;color:var(--chalk)}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.3rem;padding-bottom:6rem;max-width:1180px}
section[data-testid="stSidebar"]{background:var(--deck);border-right:1px solid var(--hull)}
section[data-testid="stSidebar"] .block-container{padding-top:1.5rem}

.brand{font-family:var(--display);font-weight:700;font-size:2.1rem;letter-spacing:-.03em;line-height:1;color:var(--chalk)}
.brand-tag{color:var(--fog);font-size:.92rem;margin:.25rem 0 1rem}

.hero{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin:.1rem 0 .9rem}
.who{display:flex;align-items:center;gap:.9rem}
.avatar{width:54px;height:54px;border-radius:50%;background:var(--tide);color:var(--night);display:grid;place-items:center;
font-family:var(--display);font-weight:700;font-size:1.2rem;flex:none}
.who h2{margin:0;padding:0;font-size:1.75rem;line-height:1.1}
.who p{margin:.2rem 0 0;color:var(--fog)}
.today{text-align:right;color:var(--fog);font-size:.92rem}
.today b{display:block;color:var(--chalk);font-family:var(--display);font-size:1.3rem;font-weight:600}

[class*="st-key-person_"] button{padding:.3rem .95rem;min-height:0}

.ribbon-card{background:var(--deck);border:1px solid var(--hull);border-radius:var(--r-lg);padding:1rem 1.15rem 1rem;margin:.2rem 0 1rem}
.ribbon-head{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin-bottom:.7rem}
.ribbon-head b{font-family:var(--display);font-size:1.1rem;font-weight:600}
.ribbon-head span{color:var(--fog);font-size:.88rem}
.ribbon{display:flex;gap:.5rem;align-items:flex-start}
.gutter{position:relative;width:2.5rem;flex:none;margin-top:2.35rem}
.gutter span{position:absolute;right:0;transform:translateY(-50%);font-size:.72rem;color:var(--fog);font-variant-numeric:tabular-nums}
.day{flex:1;min-width:30px;display:flex;flex-direction:column;gap:.35rem}
.dlabel{height:2rem;display:flex;flex-direction:column;justify-content:center;align-items:center;line-height:1.15;color:var(--fog);font-size:.76rem}
.dlabel b{color:var(--chalk);font-weight:600;font-size:.86rem}
.day.is-today .dlabel b{color:var(--gold)}
.track{position:relative;border-radius:10px;background-color:rgba(255,255,255,.03);
background-image:linear-gradient(to bottom,rgba(255,255,255,.07) 1px,transparent 1px);overflow:hidden}
.day.is-today .track{outline:1px solid var(--gold)}
.blk{position:absolute;left:12%;right:12%;border-radius:6px;opacity:.93}
.dfoot{text-align:center;font-size:.78rem;color:var(--fog);min-height:1.1rem;font-variant-numeric:tabular-nums}
.untimed{display:flex;justify-content:center;gap:4px;flex-wrap:wrap;min-height:10px}
.udot{width:9px;height:9px;border-radius:50%}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin-top:.85rem;color:var(--fog);font-size:.86rem}
.legend i{display:inline-block;width:.65rem;height:.65rem;border-radius:3px;margin-right:.4rem}
.ribbon-empty{color:var(--fog);padding:.4rem 0}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.7rem;margin:.5rem 0 1rem}
.stat{background:var(--deck);border:1px solid var(--hull);border-radius:var(--r-md);padding:.7rem .95rem}
.stat b{display:block;font-family:var(--display);font-size:1.7rem;line-height:1.1;font-weight:600;font-variant-numeric:tabular-nums}
.stat span{color:var(--fog);font-size:.86rem}

.logday{margin:1.2rem 0 .3rem;display:flex;align-items:baseline;gap:.7rem}
.logday .wd{font-family:var(--display);font-size:1.2rem;font-weight:600}
.logday .dt{color:var(--fog)}
.logday .hrs{margin-left:auto;color:var(--fog);font-variant-numeric:tabular-nums}
.entry{border-left:3px solid var(--c);background:var(--deck);border-radius:0 var(--r-md) var(--r-md) 0;padding:.7rem 1rem;margin:.45rem 0}
.entry .proj{display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap;font-weight:600}
.entry .meta{color:var(--fog);font-weight:400;font-size:.85rem;font-variant-numeric:tabular-nums}
.entry ul{margin:.4rem 0 0 1.1rem;padding:0}
.entry li{margin:.15rem 0}
.entry .meet{margin-top:.45rem;color:var(--fog);font-size:.9rem}

.welcome{background:var(--deck);border:1px solid var(--hull);border-radius:var(--r-lg);padding:1.3rem 1.5rem;margin:.8rem 0}
.welcome h3{margin:0 0 .45rem;font-size:1.45rem}
.welcome p{color:var(--fog);margin:.25rem 0}
.welcome code,.hint code{background:rgba(255,255,255,.08);padding:.1rem .4rem;border-radius:6px;color:var(--chalk)}
.hint{color:var(--fog);font-size:.92rem;margin:.2rem 0 .8rem}

[data-testid="stChatMessage"]{background:transparent;padding:.35rem 0;gap:0}
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{display:none}
[data-testid="stChatMessageContent"]{width:fit-content}
[data-testid="stChatMessageContent"] h1{font-size:1.45rem}
[data-testid="stChatMessageContent"] h2{font-size:1.2rem}
[data-testid="stChatMessageContent"] h3{font-size:1.05rem}
[data-testid="stChatMessageContent"]{background:var(--deck);border:1px solid var(--hull);border-radius:14px;padding:.7rem 1rem;max-width:min(46rem,92%)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{background:rgba(79,209,197,.12);border-color:rgba(79,209,197,.45)}
[data-testid="stChatInput"]{border-radius:16px;border:1px solid var(--hull);background:var(--deck)}

.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;border:1px solid var(--hull);
background:var(--deck2);color:var(--chalk);font-weight:500;transition:border-color .15s,background .15s}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--tide);color:var(--chalk)}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--tide);border-color:var(--tide);color:var(--night);font-weight:600}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}

[data-baseweb="tab-list"]{gap:.5rem;border-bottom:1px solid var(--hull)}
[data-baseweb="tab"]{font-weight:500}
[data-baseweb="tab"] p,[data-baseweb="tab"] span{color:var(--fog)!important}
[data-baseweb="tab"][aria-selected="true"] p,[data-baseweb="tab"][aria-selected="true"] span{color:var(--chalk)!important}
[data-baseweb="tab-highlight"]{background:var(--gold)!important}
[data-baseweb="tab-border"]{background:var(--hull)!important}
[data-testid="stExpander"]{border:1px solid var(--hull);border-radius:var(--r-md);background:transparent}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:var(--chalk)!important}

/* form controls, bottom bar and tables follow the dark theme even if .streamlit/config.toml is not loaded */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:var(--night)!important}
[data-testid="stChatInput"]>div,[data-testid="stChatInput"] textarea{background:var(--deck)!important;color:var(--chalk)!important}
[data-testid="stChatInput"] textarea::placeholder{color:var(--fog)!important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="input"]>div,[data-baseweb="base-input"]{background:var(--deck2)!important;color:var(--chalk)!important;border-color:var(--hull)!important}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:var(--chalk)!important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--deck2)!important}
[data-baseweb="popover"] li{color:var(--chalk)!important}
label,[data-testid="stWidgetLabel"] p{color:var(--fog)!important}
[data-testid="stDataFrame"]{border:1px solid var(--hull);border-radius:var(--r-md);overflow:hidden;filter:invert(.92) hue-rotate(180deg)}
[data-testid="stAlert"]{background:var(--deck2);border:1px solid var(--hull);color:var(--chalk)}
[role="tabpanel"]:focus,[data-baseweb="tab-panel"]:focus{outline:none!important;box-shadow:none!important}
.stAppDeployButton,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}

.st-key-paper{background:var(--paper);border-radius:var(--r-lg);padding:1.4rem 1.7rem}
.st-key-paper,.st-key-paper *{color:var(--ink)!important}
.st-key-paper h1{font-size:1.9rem}.st-key-paper h2{font-size:1.3rem}.st-key-paper table{border-collapse:collapse}
.st-key-paper th,.st-key-paper td{border:1px solid #cfd8e6;padding:.35rem .6rem}

@media (max-width:640px){.who h2{font-size:1.35rem}.gutter{display:none}.today{text-align:left}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""


# ------------------------------------------------------------------ helpers
def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def build_color_map(logs: List[Dict[str, Any]]) -> Dict[str, str]:
    names = sorted({l["project_name"] for l in logs if l.get("project_name")}, key=str.lower)
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _hour_label(h: int) -> str:
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def hero_html(profile: Dict[str, Any], today: Optional[date] = None) -> str:
    today = today or date.today()
    sub = " at ".join(x for x in (profile.get("role"), profile.get("company")) if x) or "No role added yet"
    return (
        '<div class="hero"><div class="who">'
        f'<div class="avatar">{esc(initials(profile["name"]))}</div>'
        f'<div><h2>{esc(profile["name"])}</h2><p>{esc(sub)}</p></div></div>'
        f'<div class="today"><b>{today.strftime("%A")}</b>{today.strftime("%d %B %Y")}</div></div>'
    )


def stat_strip(items: List[tuple]) -> str:
    cells = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>' for label, v in items)
    return f'<div class="stats">{cells}</div>'


def ribbon_card(
    logs: List[Dict[str, Any]],
    cmap: Dict[str, str],
    title: str = "This week",
    days: int = 7,
    today: Optional[date] = None,
    height: int = 150,
) -> str:
    """The signature element: one column per day, one coloured block per timed update."""
    today = today or date.today()
    day_list = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    by_day: Dict[str, List[Dict[str, Any]]] = {d.isoformat(): [] for d in day_list}
    for l in logs:
        if l.get("date") in by_day:
            by_day[l["date"]].append(l)

    spans = []
    for items in by_day.values():
        for l in items:
            a, b = parse_time_minutes(l.get("time_in")), parse_time_minutes(l.get("time_out"))
            if a is not None and b is not None and b > a:
                spans.append((a, b))
    start_h = min(6, min(a for a, _ in spans) // 60) if spans else 6
    end_h = max(20, -(-max(b for _, b in spans) // 60)) if spans else 20
    span = end_h - start_h

    total_updates = sum(len(v) for v in by_day.values())
    if not total_updates:
        return (
            f'<div class="ribbon-card"><div class="ribbon-head"><b>{esc(title)}</b></div>'
            '<div class="ribbon-empty">No updates in this period yet. Tell me what you worked on and it will show up here.</div></div>'
        )

    gutter = "".join(
        f'<span style="top:{(h - start_h) / span * 100:.2f}%">{_hour_label(h)}</span>'
        for h in range(start_h, end_h + 1)
        if h % 3 == 0 and start_h < h < end_h
    )
    cols, seen, week_hours = [], [], 0.0
    for d in day_list:
        items = by_day[d.isoformat()]
        blocks, dots, hrs = [], [], 0.0
        for l in items:
            color = cmap.get(l["project_name"], PALETTE[0])
            if l["project_name"] not in seen:
                seen.append(l["project_name"])
            tip = esc(f'{l["project_name"]}: ' + "; ".join(l["tasks"]))
            a, b = parse_time_minutes(l.get("time_in")), parse_time_minutes(l.get("time_out"))
            if a is not None and b is not None and b > a:
                top = (a / 60 - start_h) / span * 100
                blocks.append(
                    f'<i class="blk" title="{tip}" style="top:{top:.2f}%;height:{(b - a) / 60 / span * 100:.2f}%;background:{color}"></i>'
                )
                hrs += hours_worked(l) or 0.0
            else:
                dots.append(f'<i class="udot" title="{tip}" style="background:{color}"></i>')
        week_hours += hrs
        cls = "day is-today" if d == today else "day"
        cols.append(
            f'<div class="{cls}"><div class="dlabel"><b>{d.strftime("%a")}</b><span>{d.day}</span></div>'
            f'<div class="track" style="height:{height}px;background-size:100% {100 / span:.4f}%">{"".join(blocks)}</div>'
            f'<div class="dfoot">{f"{hrs:.1f}h" if hrs else ""}</div><div class="untimed">{"".join(dots)}</div></div>'
        )
    legend = "".join(
        f'<span><i style="background:{cmap.get(n, PALETTE[0])}"></i>{esc(n)}</span>' for n in seen
    )
    summary = f"{total_updates} update{'s' if total_updates != 1 else ''}" + (f" · {week_hours:.1f} h tracked" if week_hours else "")
    return (
        f'<div class="ribbon-card"><div class="ribbon-head"><b>{esc(title)}</b><span>{esc(summary)}</span></div>'
        f'<div class="ribbon"><div class="gutter" style="height:{height}px">{gutter}</div>{"".join(cols)}</div>'
        f'<div class="legend">{legend}</div></div>'
    )


def entries_html(logs: List[Dict[str, Any]], cmap: Dict[str, str], limit: int = 60) -> str:
    """Newest day first; every update shows its log number so you can say 'delete log #5'."""
    shown = sorted(logs, key=lambda l: (l["date"], l["id"]), reverse=True)[:limit]
    out, current = [], None
    day_hours: Dict[str, float] = {}
    for l in logs:
        day_hours[l["date"]] = day_hours.get(l["date"], 0.0) + (hours_worked(l) or 0.0)
    for l in shown:
        if l["date"] != current:
            current = l["date"]
            try:
                d = datetime.strptime(l["date"], "%Y-%m-%d")
                wd, dt = d.strftime("%A"), d.strftime("%d %B %Y")
            except ValueError:
                wd, dt = l["date"], ""
            hrs = f'{day_hours[current]:.1f} h' if day_hours.get(current) else ""
            out.append(f'<div class="logday"><span class="wd">{esc(wd)}</span><span class="dt">{esc(dt)}</span><span class="hrs">{hrs}</span></div>')
        color = cmap.get(l["project_name"], PALETTE[0])
        when = f'{esc(l["time_in"])}–{esc(l["time_out"])}' if l.get("time_in") and l.get("time_out") else ""
        tasks = "".join(f"<li>{esc(t)}</li>" for t in l["tasks"])
        meet = f'<div class="meet">Meetings: {esc(l["meetings"])}</div>' if l.get("meetings") else ""
        out.append(
            f'<div class="entry" style="--c:{color}"><div class="proj">{esc(l["project_name"])}'
            f'<span class="meta">log #{l["id"]}</span><span class="meta">{when}</span></div><ul>{tasks}</ul>{meet}</div>'
        )
    if len(logs) > limit:
        out.append(f'<p class="hint">Showing the latest {limit} of {len(logs)} updates. Use the table view to see all.</p>')
    return "".join(out)
