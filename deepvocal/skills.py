import json

from .knowledge import EXERCISES
from .storage import dump


class LazyRegistry:
    def __init__(self, store):
        self.store = store
        self.cache = {}
        with store.connect() as db:
            for issue, (_, title, instructions) in EXERCISES.items():
                db.execute("INSERT OR IGNORE INTO skills VALUES (?,1,'active',?,?)", (
                    issue, dump({"name": issue, "description": title, "triggers": [issue]}), instructions,
                ))

    def list(self, state=None):
        with self.store.connect() as db:
            rows = db.execute("""
                SELECT name,version,state,metadata FROM skills WHERE (? IS NULL OR state=?)
                ORDER BY name,version DESC
            """, (state, state)).fetchall()
        return [{**json.loads(row["metadata"]), "version": row["version"], "state": row["state"]} for row in rows]

    def load(self, name):
        with self.store.connect() as db:
            row = db.execute("""
                SELECT version FROM skills WHERE name=? AND state='active' ORDER BY version DESC LIMIT 1
            """, (name,)).fetchone()
            key = (name, row["version"])
            if key not in self.cache:
                body = db.execute("SELECT instructions FROM skills WHERE name=? AND version=?", key).fetchone()
                self.cache[key] = body["instructions"]
        return key[1], self.cache[key]

    def propose(self, name):
        traces = self.store.trace_sample(name)
        if len(traces) < 2:
            raise ValueError("至少需要两条采用该练习的任务轨迹。")
        _, title, base = EXERCISES[name]
        now = {"name": name, "description": title, "triggers": [name], "trace_ids": traces}
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            draft = db.execute("SELECT version FROM skills WHERE name=? AND state='draft'", (name,)).fetchone()
            if draft:
                return {"name": name, "version": draft["version"], "state": "draft"}
            version = db.execute("SELECT MAX(version)+1 FROM skills WHERE name=?", (name,)).fetchone()[0]
            db.execute("INSERT INTO skills VALUES (?,?,'draft',?,?)", (
                name, version, dump(now), base + " 结束时录制同一小节，并与练习前比较。",
            ))
        return {"name": name, "version": version, "state": "draft"}

    def draft(self, name, version):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM skills WHERE name=? AND version=?", (name, version)).fetchone()
        if row:
            return {**json.loads(row["metadata"]), "version": version,
                    "state": row["state"], "instructions": row["instructions"]}
        return None

    def review(self, name, version, approve):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM skills WHERE name=? AND version=?", (name, version)).fetchone()
            if not row or row["state"] != "draft":
                raise ValueError("只能评审待处理的候选 Skill。")
            if approve:
                db.execute("UPDATE skills SET state='superseded' WHERE name=? AND state='active'", (name,))
            state = "active" if approve else "rejected"
            db.execute("UPDATE skills SET state=? WHERE name=? AND version=?", (state, name, version))
        return {"name": name, "version": version, "state": state}
