from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
from unicodedata import normalize
from uuid import uuid4


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def normalize_song_title(title):
    return "".join(char for char in normalize("NFKC", title).casefold() if char.isalnum())


def reference_title_keys(filename, song_title=None):
    if song_title:
        return {normalize_song_title(song_title)} - {""}
    name = Path(filename.replace("\\", "/")).stem
    title = re.split(r"[-–—]", name, maxsplit=1)[0]
    return {normalize_song_title(name), normalize_song_title(title)} - {""}


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
CREATE TABLE IF NOT EXISTS reference_titles (
  title_key TEXT NOT NULL, audio_id TEXT NOT NULL REFERENCES audio(id),
  PRIMARY KEY(title_key, audio_id)
);
CREATE INDEX IF NOT EXISTS reference_titles_audio ON reference_titles(audio_id);
CREATE TABLE IF NOT EXISTS reference_features (
  audio_id TEXT PRIMARY KEY REFERENCES audio(id), version INTEGER NOT NULL,
  duration REAL NOT NULL, track BLOB NOT NULL, digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_motifs (
  motif INTEGER NOT NULL, audio_id TEXT NOT NULL REFERENCES reference_features(audio_id),
  PRIMARY KEY(motif, audio_id)
);
CREATE INDEX IF NOT EXISTS reference_motifs_audio ON reference_motifs(audio_id);
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
CREATE TABLE IF NOT EXISTS memory_topics (
  id TEXT PRIMARY KEY, melody_id TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL, anchors TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_observations (
  report_id TEXT PRIMARY KEY, topic_id TEXT NOT NULL REFERENCES memory_topics(id),
  audio_id TEXT NOT NULL, observed REAL NOT NULL, bpm INTEGER NOT NULL,
  transpose INTEGER NOT NULL, offset_seconds REAL NOT NULL,
  scoring_version INTEGER NOT NULL, value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_topic_history
  ON memory_observations(topic_id, observed DESC, report_id DESC);
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
            legacy = db.execute("""
                SELECT id, filename FROM audio WHERE kind='reference' AND NOT EXISTS (
                    SELECT 1 FROM reference_titles WHERE audio_id=audio.id
                )
            """).fetchall()
            db.executemany("INSERT OR IGNORE INTO reference_titles VALUES (?,?)", (
                (title, row["id"]) for row in legacy for title in reference_title_keys(row["filename"])
            ))

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
            db.execute("UPDATE jobs SET state='queued', result=NULL WHERE state='running'")

    def claim(self):
        with self.connect() as db:
            return db.execute("""
                UPDATE jobs SET state='running' WHERE id=(
                    SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1
                ) RETURNING id, kind, payload
            """).fetchone()

    def finish(self, job, result=None, error=None, *, db=None):
        if db is None:
            with self.connect() as conn:
                return self.finish(job, result, error, db=conn)
        db.execute("UPDATE jobs SET state=?, result=?, error=? WHERE id=?", (
            "failed" if error else "succeeded", dump(result) if result is not None else None,
            error, job["id"],
        ))
        if (job["kind"] == "assessment" and result and not result.get("source_assessment")
                and result.get("melody_id") and result.get("notes")
                and result["issue"] != "insufficient_audio"):
            self.enqueue("memory", {
                "category": "assessment", "subject": result["melody_id"], "source": job["id"],
                "observed": result["observed_at"],
                "value": {k: result[k] for k in ("issue", "median_pitch_error_cents", "matched_notes")},
            }, db)

    def update_job_progress(self, key, progress):
        with self.connect() as db:
            db.execute("UPDATE jobs SET result=? WHERE id=? AND state='running'", (dump(progress), key))

    def job(self, key):
        with self.connect() as db:
            row = db.execute("""
                SELECT id,kind,state,result,error FROM jobs WHERE id=? AND kind <> 'memory'
            """, (key,)).fetchone()
        if row:
            return {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
        return None

    def planning_reports(self, *, session=None, song_title=None, segment_id=None, melody_id=None, unusable=()):
        """Read saved quantitative and qualitative assessments using the same source records."""
        title_key = normalize_song_title(song_title) if song_title else None
        with self.connect() as db:
            db.create_function("reference_matches", 2, lambda title, key: int(
                bool(title and key in reference_title_keys(title))), deterministic=True)
            rows = db.execute("""
                SELECT j.id, j.result FROM jobs j
                WHERE j.kind='assessment' AND j.state='succeeded'
                  AND json_extract(j.result, '$.issue') NOT IN (SELECT value FROM json_each(?))
                  AND (? IS NULL OR EXISTS (
                    SELECT 1 FROM turns t WHERE t.session=? AND (
                      json_extract(t.result, '$.job.id')=j.id OR EXISTS (
                        SELECT 1 FROM json_each(t.result, '$.jobs') listed
                        WHERE json_extract(listed.value, '$.id')=j.id
                      )
                    )
                  ))
                  AND (? IS NULL OR json_extract(j.result, '$.reference.segment_id')=?)
                  AND (? IS NULL OR (json_extract(j.result, '$.melody_id')=?
                                      AND json_extract(j.result, '$.reference') IS NULL))
                  AND (? IS NULL OR reference_matches(json_extract(j.result, '$.reference.title'), ?) OR EXISTS (
                    SELECT 1 FROM reference_titles r WHERE r.title_key=?
                    AND r.audio_id=json_extract(j.result, '$.reference.audio_id')
                  ))
                ORDER BY j.created DESC, j.id DESC
                LIMIT 12
                """, (dump(sorted(unusable)), session, session, segment_id, segment_id, melody_id, melody_id,
                      title_key, title_key, title_key)).fetchall()
        return [(row["id"], json.loads(row["result"])) for row in rows]

    def audio(self, key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM audio WHERE id=?", (key,)).fetchone()
        return dict(row) if row else None

    def put_audio(self, key, filename, kind, data, *, song_title=None):
        target = self.media / f"{key}.wav"
        pending = self.media / f"{key}.tmp"
        pending.write_bytes(data)
        pending.replace(target)
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO audio VALUES (?,?,?)", (key, filename, kind))
            if inserted.rowcount and kind == "reference":
                db.executemany("INSERT INTO reference_titles VALUES (?,?)",
                               ((title, key) for title in reference_title_keys(filename, song_title)))
        return key

    def unindexed_references(self, version, *, title_key=None):
        title_filter = "" if title_key is None else "AND audio.id IN (SELECT audio_id FROM reference_titles WHERE title_key=?)"
        params = (version,) if title_key is None else (version, title_key)
        with self.connect() as db:
            return [row["id"] for row in db.execute(f"""
                SELECT audio.id FROM audio LEFT JOIN reference_features
                  ON audio.id=reference_features.audio_id AND reference_features.version=?
                WHERE audio.kind='reference' AND reference_features.audio_id IS NULL
                {title_filter}
            """, params)]

    def put_reference_features(self, key, version, duration, track, digest, motifs):
        with self.connect() as db:
            db.execute("DELETE FROM reference_motifs WHERE audio_id=?", (key,))
            db.execute("INSERT OR REPLACE INTO reference_features VALUES (?,?,?,?,?)",
                       (key, version, duration, track, digest))
            db.executemany("INSERT INTO reference_motifs VALUES (?,?)", ((motif, key) for motif in motifs))

    def reference_features(self, key, version):
        with self.connect() as db:
            return db.execute("SELECT duration,track FROM reference_features WHERE audio_id=? AND version=?",
                              (key, version)).fetchone()

    def song_references(self, title):
        title_key = normalize_song_title(title)
        with self.connect() as db:
            rows = db.execute("""
                SELECT audio.id, audio.filename, reference_features.digest FROM reference_titles
                JOIN audio ON audio.id=reference_titles.audio_id AND audio.kind='reference'
                LEFT JOIN reference_features ON reference_features.audio_id=audio.id
                WHERE reference_titles.title_key=? ORDER BY audio.rowid
            """, (title_key,)).fetchall()
        unique = {}
        for row in rows:
            if (self.media / f"{row['id']}.wav").is_file():
                unique.setdefault(row["digest"] or row["id"], dict(row))
        return list(unique.values())

    def search_reference_motifs(self, motifs, version, limit, *, title_key=None):
        signatures = dump(sorted(motifs))
        if title_key is None:
            candidate_query = """
                SELECT audio_id, COUNT(*) AS hits FROM reference_motifs
                WHERE motif IN (SELECT value FROM json_each(?)) GROUP BY audio_id HAVING hits >= 2
            """
            params = (signatures,)
        else:
            candidate_query = """
                SELECT named.audio_id, COUNT(motifs.motif) AS hits FROM reference_titles AS named
                LEFT JOIN reference_motifs AS motifs ON motifs.audio_id=named.audio_id
                    AND motifs.motif IN (SELECT value FROM json_each(?))
                WHERE named.title_key=? GROUP BY named.audio_id
            """
            params = (signatures, title_key)
        with self.connect() as db:
            return db.execute(f"""
                WITH candidates AS ({candidate_query}), matches AS (
                    SELECT features.audio_id, features.digest, audio.rowid AS upload_order, candidates.hits
                    FROM candidates
                    JOIN reference_features AS features ON features.audio_id=candidates.audio_id
                    JOIN audio ON audio.id=features.audio_id
                    WHERE features.version=?
                )
                SELECT audio_id, hits FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY digest ORDER BY upload_order) AS duplicate_order
                    FROM matches
                ) WHERE duplicate_order=1 ORDER BY hits DESC, upload_order LIMIT ?
            """, (*params, version, limit)).fetchall()

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

    def pending_cover(self, session):
        with self.connect() as db:
            row = db.execute("SELECT result FROM turns WHERE session=? ORDER BY rowid DESC LIMIT 1", (session,)).fetchone()
        cover = json.loads(row["result"]).get("singing_agent") if row else None
        return cover if cover and cover.get("status") == "needs_input" else None

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
                SELECT rowid AS position, id, query, created, result FROM turns
                WHERE session=? AND (? IS NULL OR rowid < ?)
                ORDER BY rowid DESC LIMIT ?
            """, (session, before, before, limit + 1)).fetchall()
            turns = [{"id": row["id"], "query": row["query"], "created_at": row["created"],
                      "result": json.loads(row["result"])} for row in reversed(rows[:limit])]
            refs = [turn["result"].get("jobs") or ([job] if (job := turn["result"].get("job")) else []) for turn in turns]
            ids = tuple(dict.fromkeys(job["id"] for entries in refs for job in entries))
            jobs = {}
            if ids:
                slots = ",".join("?" for _ in ids)
                jobs = {row["id"]: {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
                        for row in db.execute(f"SELECT id,kind,state,result,error FROM jobs WHERE kind <> 'memory' AND id IN ({slots})", ids)}
            for turn, entries in zip(turns, refs):
                snapshots = [jobs[ref["id"]] for ref in entries if ref["id"] in jobs]
                if snapshots:
                    turn.update(job=snapshots[0], jobs=snapshots)
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
