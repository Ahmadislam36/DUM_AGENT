from datetime import date, timedelta

from dum_agent.ui_theme import build_color_map, entries_html, initials, ribbon_card, stat_strip

TODAY = date(2026, 10, 15)


def log(i, d, project="LawGPT", tasks=("CI/CD",), tin="09:00", tout="17:00", meetings=None):
    return {"id": i, "date": d.isoformat(), "project_name": project, "tasks": list(tasks),
            "time_in": tin, "time_out": tout, "meetings": meetings}


def test_initials_and_colors_are_stable():
    assert initials("ahmad khan") == "AK" and initials("Wakeel") == "W" and initials("") == "?"
    logs = [log(1, TODAY, "b"), log(2, TODAY, "A")]
    assert build_color_map(logs) == build_color_map(list(reversed(logs)))
    assert build_color_map(logs)["A"] != build_color_map(logs)["b"]


def test_ribbon_draws_timed_blocks_and_untimed_dots():
    logs = [log(1, TODAY), log(2, TODAY - timedelta(days=1), tin=None, tout=None)]
    html = ribbon_card(logs, build_color_map(logs), today=TODAY)
    assert html.count('class="blk"') == 1 and html.count('class="udot"') == 1
    assert "day is-today" in html and "8.0h" in html and "2 updates" in html


def test_ribbon_axis_grows_for_early_and_late_work_and_ignores_old_logs():
    logs = [log(1, TODAY, tin="05:00", tout="23:30"), log(2, TODAY - timedelta(days=30))]
    html = ribbon_card(logs, build_color_map(logs), today=TODAY)
    assert html.count('class="blk"') == 1 and "1 update" in html


def test_ribbon_empty_state():
    assert "No updates in this period" in ribbon_card([], {}, today=TODAY)


def test_user_text_is_escaped():
    evil = '<script>alert(1)</script>'
    logs = [log(1, TODAY, project=evil, tasks=(evil,), meetings=evil)]
    html = ribbon_card(logs, build_color_map(logs), today=TODAY) + entries_html(logs, build_color_map(logs))
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_entries_group_by_day_newest_first_and_show_log_numbers():
    logs = [log(1, TODAY - timedelta(days=2)), log(2, TODAY), log(3, TODAY, project="Other")]
    html = entries_html(logs, build_color_map(logs))
    assert html.index("log #2") < html.index("log #1") and "log #3" in html
    assert html.count('class="logday"') == 2
    assert "Showing the latest" in entries_html(logs, build_color_map(logs), limit=2)


def test_stat_strip():
    assert stat_strip([("Updates", 3)]).count('class="stat"') == 1
