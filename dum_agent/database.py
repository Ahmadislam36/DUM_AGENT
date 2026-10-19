import json
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .logger import get_logger

_log = get_logger("db")
_init_logged = False


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------- #
# Schema + migration
# --------------------------------------------------------------------------- #
def init_db() -> None:
    global _init_logged
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time_in TEXT,
                time_out TEXT,
                project_name TEXT,
                tasks TEXT,
                meetings TEXT,
                raw_input TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                role TEXT,
                company TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(daily_logs)")}
        if "profile_id" not in cols:
            conn.execute("ALTER TABLE daily_logs ADD COLUMN profile_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_logs_date ON daily_logs(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_logs_profile ON daily_logs(profile_id)")
        _migrate_legacy(conn)
    (_log.debug if _init_logged else _log.info)(
        "Database ready: %s (tables: daily_logs, profiles)", config.DB_PATH
    )
    _init_logged = True


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    """Old single-profile data -> profiles table; old logs without a profile -> first profile."""
    has_profiles = conn.execute("SELECT 1 FROM profiles LIMIT 1").fetchone()
    if not has_profiles:
        legacy = None
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_profile'"
        ).fetchone():
            legacy = conn.execute("SELECT name, role, company FROM user_profile WHERE id = 1").fetchone()
        if legacy and legacy["name"]:
            conn.execute(
                "INSERT INTO profiles (name, role, company) VALUES (?, ?, ?)",
                (legacy["name"], legacy["role"], legacy["company"]),
            )
            _log.info("Migration: old single profile %r moved to the profiles table", legacy["name"])

    orphans = conn.execute("SELECT COUNT(*) FROM daily_logs WHERE profile_id IS NULL").fetchone()[0]
    if orphans:
        row = conn.execute("SELECT id FROM profiles ORDER BY id LIMIT 1").fetchone()
        if row:
            pid = row["id"]
        else:
            pid = conn.execute("INSERT INTO profiles (name) VALUES ('Default')").lastrowid
        _log.info("Migration: assigning %d old log(s) to profile id %s", orphans, pid)
        conn.execute("UPDATE daily_logs SET profile_id = ? WHERE profile_id IS NULL", (pid,))


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def list_profiles() -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT id, name, role, company FROM profiles ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def get_profile(profile_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if profile_id is None:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, name, role, company FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
    return dict(row) if row else None


def find_profile_by_name(name: str) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, name, role, company FROM profiles WHERE name = ? COLLATE NOCASE", (name.strip(),)
        ).fetchone()
    return dict(row) if row else None


def upsert_profile(
    name: str, role: Optional[str] = None, company: Optional[str] = None
) -> Tuple[Dict[str, Any], bool]:
    """Create the profile if the name is new (case-insensitive), else update role/company.
    Returns (profile, created)."""
    name = name.strip()
    existing = find_profile_by_name(name)
    with closing(_connect()) as conn, conn:
        if existing:
            conn.execute(
                "UPDATE profiles SET role = COALESCE(?, role), company = COALESCE(?, company) WHERE id = ?",
                (role, company, existing["id"]),
            )
            pid, created = existing["id"], False
        else:
            pid = conn.execute(
                "INSERT INTO profiles (name, role, company) VALUES (?, ?, ?)", (name, role, company)
            ).lastrowid
            created = True
    _log.info("Profile %s: %r (id=%s)", "created" if created else "updated", name, pid)
    return get_profile(pid), created


def update_profile_fields(
    profile_id: int, role: Optional[str] = None, company: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE profiles SET role = COALESCE(?, role), company = COALESCE(?, company) WHERE id = ?",
            (role, company, profile_id),
        )
    _log.info("Profile id=%s fields updated (role=%r, company=%r)", profile_id, role, company)
    return get_profile(profile_id)


# --------------------------------------------------------------------------- #
# Logs (always scoped to a profile)
# --------------------------------------------------------------------------- #
def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    try:
        d["tasks"] = json.loads(d.get("tasks") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["tasks"] = [d.get("tasks")] if d.get("tasks") else []
    return d


def save_log(log_: Dict[str, Any], raw_input: str, profile_id: int) -> int:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            """
            INSERT INTO daily_logs
                (date, time_in, time_out, project_name, tasks, meetings, raw_input, profile_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_["date"],
                log_.get("time_in"),
                log_.get("time_out"),
                log_["project_name"],
                json.dumps(log_["tasks_completed"], ensure_ascii=False),
                log_.get("meetings"),
                raw_input,
                profile_id,
            ),
        )
        new_id = cur.lastrowid
    _log.info("INSERT daily_logs -> log #%s (profile_id=%s, project=%r)", new_id, profile_id, log_["project_name"])
    return new_id


def fetch_30_days_logs(profile_id: int) -> List[Dict[str, Any]]:
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM daily_logs WHERE profile_id = ? AND date >= ? ORDER BY date ASC, id ASC",
            (profile_id, cutoff),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def fetch_all_logs(profile_id: int) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM daily_logs WHERE profile_id = ? ORDER BY date ASC, id ASC", (profile_id,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def search_logs(
    profile_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if profile_id is not None:
        clauses.append("profile_id = ?")
        params.append(profile_id)
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    kws = [k.strip() for k in (keywords or []) if k and k.strip()]
    if kws:
        kw_clauses = []
        for k in kws:
            kw_clauses.append(
                "(project_name LIKE ? OR tasks LIKE ? OR meetings LIKE ? OR raw_input LIKE ?)"
            )
            params.extend([f"%{k}%"] * 4)
        clauses.append("(" + " OR ".join(kw_clauses) + ")")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM daily_logs {where} ORDER BY date DESC, id DESC LIMIT ?"
    params.append(limit)
    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in reversed(rows)]  # chronological


# --------------------------------------------------------------------------- #
# Delete / edit (always scoped to a profile)
# --------------------------------------------------------------------------- #
def count_logs(profile_id: int) -> int:
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM daily_logs WHERE profile_id = ?", (profile_id,)).fetchone()[0]


def delete_profile(profile_id: int) -> int:
    """Delete a profile and ALL its logs. Returns the number of logs deleted."""
    with closing(_connect()) as conn, conn:
        n = conn.execute("DELETE FROM daily_logs WHERE profile_id = ?", (profile_id,)).rowcount
        conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    _log.info("DELETE profile id=%s and its %d log(s)", profile_id, n)
    return n


def get_log(profile_id: int, log_id: int) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM daily_logs WHERE id = ? AND profile_id = ?", (log_id, profile_id)
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_log(profile_id: int, log_id: int) -> bool:
    with closing(_connect()) as conn, conn:
        deleted = conn.execute(
            "DELETE FROM daily_logs WHERE id = ? AND profile_id = ?", (log_id, profile_id)
        ).rowcount > 0
    if deleted:
        _log.info("DELETE daily_logs -> log #%s (profile_id=%s)", log_id, profile_id)
    return deleted


def remove_task(profile_id: int, log_id: int, task_index: int, expected_text: Optional[str] = None) -> str:
    """Remove one task from a log. Returns 'task_deleted', 'log_deleted' (it was the last task) or 'not_found'."""
    log = get_log(profile_id, log_id)
    if not log:
        return "not_found"
    tasks = log["tasks"]
    if not (0 <= task_index < len(tasks)) or (expected_text is not None and tasks[task_index] != expected_text):
        return "not_found"
    removed = tasks.pop(task_index)
    if not tasks:
        _log.info("Removed last task %r of log #%s -> deleting the whole log", removed, log_id)
        delete_log(profile_id, log_id)
        return "log_deleted"
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE daily_logs SET tasks = ? WHERE id = ? AND profile_id = ?",
            (json.dumps(tasks, ensure_ascii=False), log_id, profile_id),
        )
    _log.info("UPDATE daily_logs -> removed task %r from log #%s", removed, log_id)
    return "task_deleted"


def update_log(
    profile_id: int,
    log_id: int,
    *,
    task_index: Optional[int] = None,
    new_task_text: Optional[str] = None,
    expected_text: Optional[str] = None,
    project_name: Optional[str] = None,
    time_in: Optional[str] = None,
    time_out: Optional[str] = None,
    meetings: Optional[str] = None,
    date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Edit a log (and optionally one task in it). Only non-None values change. None if not found."""
    log = get_log(profile_id, log_id)
    if not log:
        return None
    tasks = list(log["tasks"])
    if new_task_text is not None:
        if task_index is None or not (0 <= task_index < len(tasks)):
            return None
        if expected_text is not None and tasks[task_index] != expected_text:
            return None
        tasks[task_index] = new_task_text
    new = {
        "date": date or log["date"],
        "time_in": time_in if time_in is not None else log["time_in"],
        "time_out": time_out if time_out is not None else log["time_out"],
        "project_name": project_name or log["project_name"],
        "meetings": meetings if meetings is not None else log["meetings"],
    }
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            UPDATE daily_logs SET date = ?, time_in = ?, time_out = ?, project_name = ?,
                meetings = ?, tasks = ? WHERE id = ? AND profile_id = ?
            """,
            (new["date"], new["time_in"], new["time_out"], new["project_name"], new["meetings"],
             json.dumps(tasks, ensure_ascii=False), log_id, profile_id),
        )
    _log.info("UPDATE daily_logs -> log #%s edited (profile_id=%s)", log_id, profile_id)
    return get_log(profile_id, log_id)
