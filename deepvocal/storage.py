from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY, session TEXT NOT NULL, query TEXT NOT NULL,
  result TEXT NOT NULL, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session, created);
CREATE TABLE IF NOT EXISTS deleted_sessions (session TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS audio (
  id TEXT PRIMARY KEY, filename TEXT NOT NULL, kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_events (
  id TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
  state TEXT NOT NULL, created REAL NOT NULL, result TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created);
CREATE TABLE IF NOT EXISTS facts (
  category TEXT NOT NULL, subject TEXT NOT NULL, value TEXT NOT NULL,
  source TEXT NOT NULL, observed REAL NOT NULL,
  PRIMARY KEY(category, subject)
);
CREATE TABLE IF NOT EXISTS skills (
  name TEXT NOT NULL, version INTEGER NOT NULL, state TEXT NOT NULL,
  metadata TEXT NOT NULL, instructions TEXT NOT NULL,
  PRIMARY KEY(name, version)
);
CREATE TABLE IF NOT EXISTS model_settings (
  id INTEGER PRIMARY KEY CHECK(id=1), provider TEXT NOT NULL,
  model TEXT NOT NULL, protected_key BLOB
);
"""


class Storage:
    def __init__(self, root):
        self.root = Path(root)
        self.media = self.root / "audio"
        self.media.mkdir(exist_ok=True, parents=True)
        self.db = self.root / "deepvocal.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, kind, payload, db=None):
        if db is None:
            with self.connect() as conn:
                return self.enqueue(kind, payload, conn)
        key = uuid4().hex
        db.execute(
            "INSERT INTO jobs (id,kind,payload,state,created) VALUES (?,?,?,'queued',?)",
            (key, kind, dump(payload), time.time()),
        )
        return key

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='queued' WHERE state='running'")

    def claim(self):
        with self.connect() as db:
            return db.execute("""
                UPDATE jobs SET state='running' WHERE id=(
                    SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1
                ) RETURNING id, kind, payload
            """).fetchone()

    def finish(self, job, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?, result=?, error=? WHERE id=?", (
                "failed" if error else "succeeded", dump(result) if result is not None else None,
                error, job["id"],
            ))
            if job["kind"] == "assessment" and result and result["notes"]:
                self.enqueue("memory", {
                    "category": "assessment", "subject": result["melody_id"], "source": job["id"],
                    "observed": result["observed_at"],
                    "value": {k: result[k] for k in ("issue", "median_pitch_error_cents", "matched_notes")},
                }, db)

    def job(self, key):
        with self.connect() as db:
            row = db.execute("""
                SELECT id,kind,state,result,error FROM jobs WHERE id=? AND kind <> 'memory'
            """, (key,)).fetchone()
        if row:
            return {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
        return None

    def audio(self, key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM audio WHERE id=?", (key,)).fetchone()
        return dict(row) if row else None

    def put_audio(self, key, filename, kind, data):
        target = self.media / f"{key}.wav"
        pending = self.media / f"{key}.tmp"
        pending.write_bytes(data)
        pending.replace(target)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO audio VALUES (?,?,?)", (key, filename, kind))
        return key

    def record_turn(self, session, query, result, profile=None):
        key = uuid4().hex
        now = time.time()
        with self.connect() as db:
            inserted = db.execute("""
                INSERT INTO turns
                SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM deleted_sessions WHERE session=?)
            """, (key, session, query, dump(result), now, session))
            if not inserted.rowcount:
                return None
            for field, value in (profile or {}).items():
                self.enqueue("memory", {
                    "category": "profile", "subject": field, "value": value, "source": key, "observed": now,
                }, db)
            if result.get("evidence") and "qa" in result["steps"]:
                self.enqueue("memory", {
                    "category": "question", "subject": result["evidence"][0]["id"],
                    "value": {"question": query}, "source": key, "observed": now,
                }, db)
        return key

    def session_deleted(self, session):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM deleted_sessions WHERE session=?", (session,)).fetchone() is not None

    def delete_session(self, session):
        with self.connect() as db:
            deleted = db.execute("""
                INSERT INTO deleted_sessions
                SELECT ? WHERE EXISTS (SELECT 1 FROM turns WHERE session=?)
                ON CONFLICT(session) DO NOTHING
            """, (session, session))
            if not deleted.rowcount:
                return
            db.execute("""
                DELETE FROM facts WHERE category='question' AND source IN (SELECT id FROM turns WHERE session=?)
            """, (session,))
            db.execute("""
                DELETE FROM jobs WHERE kind='memory' AND json_extract(payload, '$.category')='question'
                    AND json_extract(payload, '$.source') IN (SELECT id FROM turns WHERE session=?)
            """, (session,))
            db.execute("DELETE FROM turns WHERE session=?", (session,))

    def history(self, session):
        with self.connect() as db:
            rows = db.execute("""
                SELECT query,result FROM turns WHERE session=? ORDER BY created DESC LIMIT 6
            """, (session,)).fetchall()
        history = []
        for row in reversed(rows):
            result = json.loads(row["result"])
            local = any(not item.get("allow_cloud", True) for item in result.get("evidence", []))
            history.append({"query": row["query"][:450], "answer": "" if local else result.get("answer", "")[:450]})
        return history

    def model_settings(self):
        with self.connect() as db:
            return db.execute("SELECT provider,model,protected_key FROM model_settings WHERE id=1").fetchone()

    def update_model_settings(self, provider, model, protected_key):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if provider == "local":
                db.execute("DELETE FROM model_settings WHERE id=1")
                return
            previous = db.execute("SELECT provider,protected_key FROM model_settings WHERE id=1").fetchone()
            if protected_key is None and previous and previous["provider"] == provider:
                protected_key = previous["protected_key"]
            if provider != "custom" and protected_key is None:
                raise ValueError("请填写该服务商的 API Key。")
            db.execute("""
                INSERT INTO model_settings VALUES (1,?,?,?)
                ON CONFLICT(id) DO UPDATE SET provider=excluded.provider,
                    model=excluded.model, protected_key=excluded.protected_key
            """, (provider, model, protected_key))

    def sessions(self, limit, offset):
        with self.connect() as db:
            rows = db.execute("""
                WITH recent AS (
                    SELECT session, COUNT(*) AS turn_count, MAX(created) AS updated_at
                    FROM turns GROUP BY session
                    ORDER BY updated_at DESC, session DESC LIMIT ? OFFSET ?
                )
                SELECT session AS id, turn_count, updated_at, (
                    SELECT substr(query, 1, 100) FROM turns
                    WHERE session=recent.session ORDER BY created, rowid LIMIT 1
                ) AS title
                FROM recent ORDER BY updated_at DESC, session DESC
            """, (limit + 1, offset)).fetchall()
        return {"sessions": [dict(row) for row in rows[:limit]],
                "next_offset": offset + limit if len(rows) > limit else None}

    def session_turns(self, session, limit, before=None):
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM turns WHERE session=? LIMIT 1", (session,)).fetchone():
                return None
            rows = db.execute("""
                SELECT t.position, t.id, t.query, t.created, t.result,
                    j.id AS job_id, j.kind AS job_kind, j.state AS job_state,
                    j.result AS job_result, j.error AS job_error
                FROM (
                    SELECT rowid AS position, id, query, created, result FROM turns
                    WHERE session=? AND (? IS NULL OR rowid < ?)
                    ORDER BY rowid DESC LIMIT ?
                ) AS t
                LEFT JOIN jobs AS j ON j.id=json_extract(t.result, '$.job.id') AND j.kind <> 'memory'
                ORDER BY t.position DESC
            """, (session, before, before, limit + 1)).fetchall()
        turns = []
        for row in reversed(rows[:limit]):
            turn = {"id": row["id"], "query": row["query"], "created_at": row["created"],
                    "result": json.loads(row["result"])}
            if row["job_id"]:
                turn["job"] = {"id": row["job_id"], "kind": row["job_kind"], "state": row["job_state"],
                               "result": json.loads(row["job_result"]) if row["job_result"] else None,
                               "error": row["job_error"]}
            turns.append(turn)
        return {"session_id": session, "turns": turns,
                "next_before": rows[limit - 1]["position"] if len(rows) > limit else None}

    def facts(self, category=None, limit=30):
        with self.connect() as db:
            rows = db.execute("""
                SELECT * FROM facts WHERE (? IS NULL OR category=?) ORDER BY observed DESC LIMIT ?
            """, (category, category, limit)).fetchall()
        return [{**dict(row), "value": json.loads(row["value"])} for row in rows]

    def store_fact(self, payload):
        with self.connect() as db:
            db.execute("""
                INSERT INTO facts
                SELECT ?,?,?,?,? WHERE ? <> 'question' OR EXISTS (SELECT 1 FROM turns WHERE id=?)
                ON CONFLICT(category,subject) DO UPDATE SET value=excluded.value,
                source=excluded.source, observed=excluded.observed
                WHERE excluded.observed >= facts.observed
            """, (payload["category"], payload["subject"], dump(payload["value"]),
                  payload["source"], payload["observed"], payload["category"], payload["source"]))

    def record_progress(self, payload):
        key = uuid4().hex
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO progress_events VALUES (?,?,?)", (key, dump(payload), now))
            self.enqueue("memory", {"category": "progress", "subject": payload["melody_id"],
                                    "source": key, "observed": now, "value": payload}, db)
        return key

    def trace_sample(self, issue):
        with self.connect() as db:
            rows = db.execute("SELECT id,result FROM turns ORDER BY created DESC LIMIT 500").fetchall()
        return [
            row["id"] for row in rows
            if (plan := json.loads(row["result"]).get("plan")) and plan["focus"] == issue
        ][:20]
