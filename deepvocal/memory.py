"""Assessment memory for the local DeepVocal profile."""

from functools import lru_cache
import json
import logging
import re
from unicodedata import normalize

import numpy as np

from .audio import MELODIES, UNUSABLE_ISSUES
from .storage import dump


ANCHORS = {
    "warmup": ("八音练习", "八音热身", "上行下行练习"),
    "triad": ("三和弦练习", "五音练习", "三和弦跟唱"),
}
SUMMARY_FIELDS = ("issue", "median_pitch_error_cents", "matched_notes", "total_notes", "within_50_cents_ratio")
log = logging.getLogger(__name__)


def topic_for(melody_id):
    title = MELODIES[melody_id]["title"]
    return {"id": f"assessment:{melody_id}", "melody_id": melody_id,
            "title": f"{title} · 测评历史", "anchors": [title, *ANCHORS[melody_id]]}


def normalize_query(text):
    return re.sub(r"\s+", "", normalize("NFKC", text).casefold())


def observation(row):
    value = {key: row[key] for key in (
        "report_id", "observed", "bpm", "transpose", "offset_seconds", "scoring_version"
    )} | json.loads(row["value"])
    if value["bpm"] == 0:
        value["bpm"] = None
    return value


class MemoryService:
    def __init__(self, store, embed=None):
        self.store = store
        self.embed = embed
        self.catalog = [topic_for(melody_id) for melody_id in MELODIES]
        self.index_texts = tuple(
            "，".join([topic["title"], *topic["anchors"]]) for topic in self.catalog
        )
        self.topic_vectors = lru_cache(maxsize=1)(embed) if embed else None

    def resolve_melody(self, query):
        """Resolve a built-in reference by ID, known aliases or one local vector search."""
        text = normalize_query(query)
        matches = [topic["melody_id"] for topic in self.catalog
                   if text == topic["melody_id"]
                   or any(normalize_query(anchor) in text for anchor in topic["anchors"])]
        if matches:
            return matches[0] if len(matches) == 1 else None
        if not text or self.embed is None:
            return None
        try:
            vectors = np.asarray(self.topic_vectors(self.index_texts))
            vector = np.asarray(self.embed([query])[0])
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(vector)
            scores = np.divide(vectors @ vector, norms, out=np.zeros(len(vectors)), where=norms > 0)
            order = np.argsort(scores)
            best, second = order[-1], order[-2]
            if scores[best] >= 0.5 and scores[best] - scores[second] >= 0.05:
                return self.catalog[best]["melody_id"]
        except Exception:
            log.warning("Local memory similarity search failed", exc_info=True)
        return None

    def assessment_rows(self, db, topic_id, limit):
        return db.execute("""
            SELECT * FROM memory_observations WHERE topic_id=?
            ORDER BY observed DESC, report_id DESC LIMIT ?
        """, (topic_id, limit)).fetchall()

    def assessment_history(self, melody_id, limit):
        return self.history_for_topic(topic_for(melody_id), limit)

    def segment_history(self, segment_id, limit):
        return self.history_for_topic({"id": f"assessment:{segment_id}", "segment_id": segment_id}, limit)

    def history_for_topic(self, topic, limit):
        with self.store.connect() as db:
            stored = db.execute("SELECT title,anchors FROM memory_topics WHERE id=?", (topic["id"],)).fetchone()
            if stored is not None:
                topic.update(title=stored["title"], anchors=json.loads(stored["anchors"]))
            elif "segment_id" in topic:
                return None
            rows = self.assessment_rows(db, topic["id"], limit)
            return {
                "topic": topic,
                "observations": [observation(row) for row in rows],
                "comparison": self.compare_assessments(db, rows[0]) if rows else None,
            }

    def latest_assessment(self, melody_id):
        with self.store.connect() as db:
            rows = self.assessment_rows(db, topic_for(melody_id)["id"], 1)
            return observation(rows[0]) if rows else None

    def compare_assessments(self, db, current):
        prior = db.execute("""
            SELECT * FROM memory_observations
            WHERE topic_id=? AND observed < ? AND audio_id <> ?
              AND bpm=? AND transpose=? AND scoring_version=?
              AND json_extract(value, '$.matched_note_numbers')=json_extract(?, '$.matched_note_numbers')
            ORDER BY observed DESC, report_id DESC LIMIT 1
        """, tuple(current[key] for key in (
            "topic_id", "observed", "audio_id", "bpm", "transpose", "scoring_version", "value"
        ))).fetchone()
        if prior is None:
            return None
        present, previous = observation(current), observation(prior)
        reduction = abs(previous["median_pitch_error_cents"]) - abs(present["median_pitch_error_cents"])
        return {"current": present, "previous": previous, "absolute_bias_reduction_cents": round(reduction, 1)}

    def record_assessment(self, db, report_id, spec, report):
        if (report.get("source_assessment") or not report.get("issue") or report["issue"] in UNUSABLE_ISSUES
                or report.get("median_pitch_error_cents") is None
                or not all(key in spec for key in ("audio_id", "bpm", "transpose", "offset_seconds"))):
            return None
        reference = report.get("reference")
        if reference and (segment_id := reference.get("segment_id")):
            scope = f"{reference['segment_start_seconds']:g}–{reference['segment_end_seconds']:g} 秒"
            topic = {"id": f"assessment:{segment_id}", "melody_id": segment_id,
                     "title": f"{reference['title']} · {scope}", "anchors": [reference["title"], scope]}
        elif not reference and spec.get("melody_id") in MELODIES:
            topic = topic_for(spec["melody_id"])
        else:
            return None
        db.execute("INSERT INTO memory_topics (id,melody_id,title,anchors) VALUES (?,?,?,?) ON CONFLICT(id) DO NOTHING",
                   (topic["id"], topic["melody_id"], topic["title"], dump(topic["anchors"])))
        fields = {key: report[key] for key in SUMMARY_FIELDS}
        if reference:
            fields["reference"] = reference
            fields["matched_note_numbers"] = [note["reference_start_seconds"] for note in report["notes"]]
        else:
            fields["matched_note_numbers"] = [note["note"] for note in report["notes"]]
        db.execute("""
            INSERT INTO memory_observations
              (report_id,topic_id,audio_id,observed,bpm,transpose,offset_seconds,scoring_version,value)
            VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(report_id) DO NOTHING
        """, (report_id, topic["id"], spec["audio_id"], report["observed_at"], 0 if reference else spec["bpm"],
              reference.get("transpose", spec["transpose"]) if reference else spec["transpose"],
              spec["offset_seconds"], report.get("scoring_version", 1), dump(fields)))
        return db.execute("SELECT * FROM memory_observations WHERE report_id=?", (report_id,)).fetchone()

    def finish_assessment(self, job, spec, report):
        with self.store.connect() as db:
            current = self.record_assessment(db, job["id"], spec, report)
            if current is not None:
                comparison = self.compare_assessments(db, current)
                if comparison is not None:
                    report["comparison"] = comparison
            self.store.finish(job, report, db=db)

    def backfill_assessments(self):
        """Index earlier successful assessments using their original request settings."""
        with self.store.connect() as db:
            rows = db.execute("""
                SELECT id, payload, result FROM jobs
                WHERE kind='assessment' AND state='succeeded'
                  AND json_extract(result, '$.issue') NOT IN (SELECT value FROM json_each(?))
                  AND json_extract(result, '$.source_assessment') IS NULL
                  AND NOT EXISTS (SELECT 1 FROM memory_observations WHERE report_id=jobs.id)
            """, (dump(sorted(UNUSABLE_ISSUES)),)).fetchall()
            for row in rows:
                self.record_assessment(db, row["id"], json.loads(row["payload"]), json.loads(row["result"]))
