"""LangGraph recording assessment and local single-melody alignment."""

from functools import lru_cache
from hashlib import sha256
import logging
from threading import Lock
import time
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from langgraph.graph import END, START, StateGraph

from . import audio
from .storage import normalize_song_title


HOP = 0.04
SONG_SCORING_VERSION = 1
SCALES = (1.0, 0.95, 1.05, 0.90, 1.10, 0.85, 1.15)
REFERENCE_INDEX_VERSION = 1
MAX_SONG_CANDIDATES = 12
LIBRARY_MAX_ERROR = 0.95
ALIGNMENT_MARGIN = 0.28
log = logging.getLogger(__name__)


def pitch_track(samples, rate):
    # Average adjacent samples before downsampling to keep the pitch search cheap.
    factor = max(1, rate // 8000)
    signal = samples[:len(samples) // factor * factor].reshape(-1, factor).mean(axis=1)
    effective_rate = round(rate / factor)
    hop = rate * HOP / factor
    half_window = round(rate * 0.025 / factor)
    frames = np.full(int(len(signal) / hop), np.nan, dtype=np.float32)
    for index in range(1, len(frames)):
        center = round(index * hop)
        if center - half_window < 0 or center + half_window > len(signal):
            continue
        frequency = audio.pitch(signal[center - half_window:center + half_window], effective_rate)
        if frequency is not None:
            frames[index] = 69 + 12 * np.log2(frequency / 440)
    return frames


def locate(reference, performance, origin, transpose, start, end, *, infer_transpose=False):
    voiced = np.flatnonzero(np.isfinite(performance))
    first, last = int(voiced[0]), int(voiced[-1])
    offsets = voiced - first
    sung = performance[voiced] - transpose
    anchors = np.linspace(0, len(sung) - 1, min(len(sung), 32), dtype=int)
    candidates = []
    for scale in SCALES:
        width = int(round((last - first) * scale)) + 1
        starts = np.arange(start, end - width + 1)
        scaled = np.rint(offsets * scale).astype(int)
        for block in range(0, len(starts), 128):
            positions = starts[block:block + 128]
            expected = reference[positions[:, None] + scaled]
            valid = np.isfinite(expected)
            delta = sung - expected
            shifts = np.zeros(len(positions), dtype=int)
            credible = valid.mean(axis=1) >= (0.82 if infer_transpose else 0.67)
            if infer_transpose:
                offsets_at_match = np.ma.masked_invalid(delta[:, anchors])
                shifts = np.rint(np.ma.median(offsets_at_match, axis=1).filled(0)).astype(int)
                credible &= np.abs(shifts) <= 12
                delta -= shifts[:, None]
            errors = np.where(valid, np.minimum(np.abs(delta), 4), 3)
            scores = errors.mean(axis=1) + 0.15 * abs(scale - 1)
            passing = np.flatnonzero(credible & (scores <= 1.35))
            for index in passing:
                left = int(positions[index])
                candidates.append((float(scores[index]), left, left + width, scale, int(shifts[index])))

    # Adjacent starts and different tempos of the same occurrence count as one candidate.
    distinct = []
    for score, left, right, scale, shift in sorted(candidates):
        if any(min(right, item["end"]) - max(left, item["start"])
               >= 0.5 * min(right - left, item["end"] - item["start"]) for item in distinct):
            continue
        distinct.append({"start": left, "end": right, "scale": scale, "error": score,
                         "origin": origin + first, "detected_transpose": shift})
        if len(distinct) == 3:
            break
    return distinct


def reference_notes(track):
    rounded = np.where(np.isfinite(track), np.rint(track), -1).astype(int)
    boundaries = np.flatnonzero(np.r_[True, rounded[1:] != rounded[:-1], True])
    return [(int(start), int(end), int(rounded[start])) for start, end in zip(boundaries, boundaries[1:])
            if end - start >= 3 and rounded[start] >= 0]


def melody_motifs(track):
    # Ignore repeated pitches and use triples of signed intervals to index four-note phrases.
    notes = []
    for _, _, midi in reference_notes(track):
        if not notes or midi != notes[-1]:
            notes.append(midi)
    intervals = np.diff(notes).tolist()
    return {((a + 24) * 49 + b + 24) * 49 + c + 24
            for a, b, c in zip(intervals, intervals[1:], intervals[2:]) if max(abs(a), abs(b), abs(c)) <= 24}


def index_song_reference(store, key, data, samples, rate):
    track = pitch_track(samples, rate)
    duration = len(samples) / rate
    store.put_reference_features(key, REFERENCE_INDEX_VERSION, duration, track.astype("<f4", copy=False).tobytes(),
                                 sha256(data).hexdigest(), melody_motifs(track))
    return track, duration


def voiced_attacks(track):
    voiced = np.isfinite(track)
    return np.flatnonzero(voiced[2:-1] & voiced[3:] & ~voiced[1:-2] & ~voiced[:-3]) + 2


def song_markers(spec, reference, match, duration):
    start = round(match["start"] * HOP, 3)
    end = round(min(duration, match["end"] * HOP), 3)
    selected_start = spec.get("reference_start_seconds")
    selected_end = spec.get("reference_end_seconds")
    start_ms = round(1000 * (start if selected_start is None else selected_start))
    end_ms = round(1000 * (end if selected_end is None else selected_end))
    segment = uuid5(NAMESPACE_URL, f"deepvocal:{spec['reference_audio_id']}:{start_ms}:{end_ms}").hex
    markers = {"audio_id": spec["reference_audio_id"], "title": reference["filename"],
               "duration_seconds": round(duration, 3), "segment_id": segment,
               "start_seconds": start, "end_seconds": end,
               "segment_start_seconds": start_ms / 1000, "segment_end_seconds": end_ms / 1000,
               "transpose": spec["transpose"], "selection": "manual" if selected_start is not None else "automatic"}
    if spec.get("library_match"):
        markers.update(selection="library", detected_transpose=match["detected_transpose"],
                       fingerprint_hits=match["fingerprint_hits"])
        if title := spec.get("song_title"):
            markers["searched_song_title"] = title
    return markers


def measure_song(spec, reference, performance, match, duration):
    start, end, origin, scale = (match[key] for key in ("start", "end", "origin", "scale"))
    notes = [(left, right, target) for left, right, target in reference_notes(reference)
             if min(right, end) - max(left, start) >= max(3, (right - left) / 2)]
    frame_map = np.arange(len(performance)) * scale + start - origin * scale
    frames = np.isfinite(performance)
    reference_attacks, performance_attacks = set(voiced_attacks(reference)), voiced_attacks(performance)
    last_attack = -1
    timing = []
    measured, all_errors = [], []
    for number, (left, right, target) in enumerate(notes, start=1):
        # Compare the sustained portion and keep transition frames out of the median.
        trim = 1 if right - left >= 5 else 0
        mask = frames & (frame_map >= max(start, left + trim)) & (frame_map < min(end, right - trim))
        if np.count_nonzero(mask) < 2:
            continue
        target_f0_midi = float(np.median(reference[left:right])) + spec["transpose"]
        cents = (performance[mask] - target_f0_midi) * 100
        all_errors.extend(cents.tolist())
        expected_start = origin + (left - start) / scale
        onset = None
        if left in reference_attacks:
            alignment_pitch = target_f0_midi + match.get("detected_transpose", 0)
            near = performance_attacks[(performance_attacks > last_attack)
                                       & (abs(performance_attacks - expected_start) <= 4)
                                       & (abs(performance[performance_attacks] - alignment_pitch) <= 2.5)]
            if len(near) == 1:
                last_attack = near[0]
                onset = round((last_attack - expected_start) * HOP * 1000)
                timing.append(onset)
        measured.append({
            "note": number, "target_midi": target + spec["transpose"],
            "start_seconds": round(max(0, expected_start) * HOP, 3),
            "reference_start_seconds": round(left * HOP, 3),
            "pitch_error_cents": round(float(np.median(cents)), 1),
            "pitch_iqr_cents": round(float(np.percentile(cents, 75) - np.percentile(cents, 25)), 1),
            "onset_error_ms": onset,
        })
    issue = "insufficient_audio"
    if len(measured) >= max(2, (len(notes) + 1) // 2):
        bias = float(np.median([note["pitch_error_cents"] for note in measured]))
        if bias < -35:
            issue = "pitch_flat"
        elif bias > 35:
            issue = "pitch_sharp"
        elif len(timing) >= 2 and np.median(np.abs(timing)) > 100:
            issue = "timing"
        elif np.median([note["pitch_iqr_cents"] for note in measured]) > 55:
            issue = "stability"
        else:
            issue = "general"
    return report_metrics(issue, measured, len(notes), all_errors) | {
        "timing_observations": len(timing),
        "alignment": {"recording_start_seconds": round(origin * HOP, 3),
                      "recording_end_seconds": round(min(len(performance), origin + (end - start) / scale) * HOP, 3),
                      "reference_duration_seconds": round(duration, 3), "relative_speed": scale},
    }


def report_metrics(issue, notes=(), total=0, errors=()):
    quality = {"insufficient_audio": "insufficient_audio", "alignment_uncertain": "needs_selection",
               "reference_unusable": "reference_unusable", "reference_not_found": "requires_reference",
               "reference_mismatch": "requires_reference", "reference_segment_mismatch": "requires_reference",
               "song_title_required": "requires_song_title"}
    return {
        "melody_id": None, "observed_at": time.time(), "scoring_version": SONG_SCORING_VERSION,
        "issue": issue, "quality": quality.get(issue, "usable"),
        "notes": list(notes), "matched_notes": len(notes), "total_notes": total,
        "median_pitch_error_cents": round(float(np.median(errors)), 1) if len(errors) else None,
        "within_50_cents_ratio": round(float(np.mean(np.abs(errors) <= 50)), 3) if len(errors) else None,
        "timing_observations": 0,
    }


class AssessmentState(TypedDict, total=False):
    spec: dict
    search_only: bool
    candidates_only: bool
    references: list[dict]
    data: bytes
    reference: dict
    duration: float
    reference_track: np.ndarray
    performance: np.ndarray
    match: dict
    result: dict


def build_assessment_graph(store):
    indexing = Lock()

    @lru_cache(maxsize=4)
    def reference_features(key):
        if row := store.reference_features(key, REFERENCE_INDEX_VERSION):
            return np.frombuffer(row["track"], dtype="<f4"), row["duration"]
        data = (store.media / f"{key}.wav").read_bytes()
        samples, rate = audio.read_wave(data, audio.MAX_REFERENCE_SECONDS)
        return index_song_reference(store, key, data, samples, rate)

    def load(state):
        key = state["spec"]["audio_id"]
        return {"data": (store.media / f"{key}.wav").read_bytes()}

    def extract(state):
        key = state["spec"]["reference_audio_id"]
        reference = store.audio(key)
        track, duration = reference_features(key)
        samples, rate = audio.read_wave(state["data"])
        return {"reference": reference, "duration": duration,
                "reference_track": track, "performance": pitch_track(samples, rate)}

    def search_reference(state):
        spec = state["spec"]
        title = spec.get("song_title")
        title_key = normalize_song_title(title) if title else None
        samples, rate = audio.read_wave(state["data"])
        performance = pitch_track(samples, rate)
        origin = round(spec["offset_seconds"] / HOP)
        useful = performance[origin:]
        missing = {"result": report_metrics("reference_not_found") | {
            "recording_audio_id": spec["audio_id"], "offset_seconds": spec["offset_seconds"],
        }}
        if title:
            missing["result"]["searched_song_title"] = title
        voiced = useful[np.isfinite(useful)]
        motifs = melody_motifs(useful)
        if (not (state.get("candidates_only") and title_key is not None)
                and (len(voiced) < 25 or len(set(np.rint(voiced))) < 3 or (title_key is None and len(motifs) < 2))):
            if title:
                missing["result"]["reference_search_reason"] = "insufficient_recording"
            return missing

        # Existing uploads are indexed on demand so upgrades can reuse their reference files.
        with indexing:
            for key in store.unindexed_references(REFERENCE_INDEX_VERSION, title_key=title_key):
                try:
                    reference_features(key)
                except (OSError, ValueError):
                    log.warning("Skipping unreadable reference audio %s", key)

        candidates = store.search_reference_motifs(motifs, REFERENCE_INDEX_VERSION, MAX_SONG_CANDIDATES + 1,
                                                    title_key=title_key)
        if title:
            missing["result"]["reference_search_reason"] = "alignment_uncertain" if candidates else "title_not_found"
        if len(candidates) > MAX_SONG_CANDIDATES and candidates[0]["hits"] == candidates[-1]["hits"]:
            return missing
        if state.get("candidates_only"):
            references = []
            for candidate in candidates[:3]:
                key = candidate["audio_id"]
                if metadata := store.audio(key):
                    references.append({"audio_id": key, "title": metadata["filename"],
                                       "fingerprint_hits": candidate["hits"]})
            return {"references": references} if references else missing
        ranked = []
        for candidate in candidates[:MAX_SONG_CANDIDATES]:
            key = candidate["audio_id"]
            track, duration = reference_features(key)
            choices = locate(track, useful, origin, 0, 0, len(track), infer_transpose=True)
            ranked.extend((match["error"], key, {**match, "fingerprint_hits": candidate["hits"]}, track, duration)
                          for match in choices)
        ranked.sort(key=lambda item: item[0])
        if (not ranked or ranked[0][0] > LIBRARY_MAX_ERROR
                or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < ALIGNMENT_MARGIN)):
            return missing
        _, key, match, track, duration = ranked[0]
        return {"spec": {**spec, "reference_audio_id": key, "library_match": True, "transpose": 0,
                         "reference_start_seconds": None, "reference_end_seconds": None},
                "reference": store.audio(key), "duration": duration, "reference_track": track,
                "performance": performance, "match": match}

    def align(state):
        spec, duration = state["spec"], state["duration"]
        key = spec["reference_audio_id"]
        reference = {"audio_id": key, "title": state["reference"]["filename"], "transpose": spec["transpose"],
                     "duration_seconds": round(duration, 3)}
        base = {"reference": reference, "recording_audio_id": spec["audio_id"],
                "offset_seconds": spec["offset_seconds"]}
        performance = state["performance"]
        origin = round(spec["offset_seconds"] / HOP)
        useful = performance[origin:]
        voiced = useful[np.isfinite(useful)]
        left, right = spec.get("reference_start_seconds"), spec.get("reference_end_seconds")
        manual = left is not None
        if manual and right > duration + 0.001:
            raise ValueError("参考片段的结束时间超出音频长度。")
        if len(voiced) < 15:
            return {"result": report_metrics("insufficient_audio") | base}
        low = round(left / HOP) if manual else 0
        track = state["reference_track"]
        high = min(len(track), round(right / HOP)) if manual else len(track)
        if np.count_nonzero(np.isfinite(track[low:high])) < 15:
            return {"result": report_metrics("reference_unusable") | base}
        choices = locate(track, useful, origin, spec["transpose"], low, high)
        if (not choices or (not manual and len(set(np.rint(voiced))) < 3)
                or (len(choices) > 1 and choices[1]["error"] - choices[0]["error"] < ALIGNMENT_MARGIN)):
            suggestions = [{"start_seconds": round(item["start"] * HOP, 3),
                            "end_seconds": round(min(duration, item["end"] * HOP), 3)} for item in choices]
            return {"result": report_metrics("alignment_uncertain") | base | {"candidates": suggestions}}
        return {"match": choices[0]}

    def measure(state):
        spec = state["spec"]
        if not spec.get("reference_audio_id"):
            if spec.get("reference_mode", "builtin") != "builtin":
                raise ValueError("请先取得与录音可靠匹配的参考歌曲。")
            return {"result": audio.assess(state["data"], spec)}
        result = measure_song(spec, state["reference_track"], state["performance"], state["match"], state["duration"])
        result["reference"] = song_markers(spec, state["reference"], state["match"], state["duration"])
        result["recording_audio_id"] = spec["audio_id"]
        return {"result": result}

    def after_load(state):
        if state.get("search_only"):
            return "search_reference"
        return "extract" if state["spec"].get("reference_audio_id") else "measure"

    graph = StateGraph(AssessmentState)
    for node in (load, extract, align, search_reference, measure):
        graph.add_node(node)
    graph.add_conditional_edges(START, lambda state: "measure" if "match" in state else "load", ["measure", "load"])
    graph.add_conditional_edges("load", after_load, ["extract", "search_reference", "measure"])
    graph.add_edge("extract", "align")
    graph.add_conditional_edges("align", lambda state: END if "result" in state else "measure", [END, "measure"])
    graph.add_edge("search_reference", END)
    graph.add_edge("measure", END)
    return graph.compile()
