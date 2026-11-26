"""Multimodal reference alignment and assessment of the original audio slices."""

from base64 import b64encode
from io import BytesIO
import json
from math import ceil
import os
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5
import wave

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from . import audio
from .assessment import report_metrics
from .model import DASHSCOPE_RERANK_BASE, Model, ModelError


DEFAULT_AUDIO_MODEL = "qwen3.8-omni-flash"
WINDOW_SECONDS = 46
WINDOW_STEP = 37
WINDOWS_PER_REQUEST = 6
CANDIDATE_INDEX_KEYS = ("index", "candidate", "candidate_index", "candidateIndex", "window_index", "windowIndex")
ReferenceMatch = Literal["matched", "mismatch", "uncertain"]
MATCH_INSTRUCTIONS = (
    "根据实际听到的主旋律和唱词判断是否为同一首歌曲的对应乐句。"
    "忽略整体移调、速度、音色和伴奏差异。只有有清晰的旋律或唱词证据支持对应时才返回 matched。"
    "只有能辨认双方演唱且有清晰证据表明是不同歌曲时才返回 mismatch。"
    "当前窗口没有找到对应乐句、声音太短、静音、伴奏盖住人声或证据冲突时返回 uncertain。"
)


class AudioOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Candidate(AudioOutput):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=1, validation_alias=AliasChoices(*CANDIDATE_INDEX_KEYS))
    start_seconds: float = Field(ge=0, validation_alias=AliasChoices("start_seconds", "startSeconds", "start"))
    end_seconds: float = Field(gt=0, validation_alias=AliasChoices("end_seconds", "endSeconds", "end"))


class Candidates(AudioOutput):
    model_config = ConfigDict(extra="ignore")

    candidates: list[Candidate] = Field(default_factory=list, max_length=8)
    match: ReferenceMatch | None = None


class MatchCheck(AudioOutput):
    match: ReferenceMatch


class Feedback(AudioOutput):
    assessable: bool
    focus: Literal["general", "pitch_flat", "pitch_sharp", "timing", "stability"]
    summary: str = Field(min_length=1, max_length=500)
    intonation: str = Field(max_length=500)
    rhythm: str = Field(max_length=500)
    stability: str = Field(max_length=500)
    exercise: str = Field(max_length=500)


def audio_model(controller):
    key = os.getenv("DEEPVOCAL_AUDIO_MODEL_API_KEY") or (controller.key if controller.provider == "dashscope" else "")
    if not key:
        raise ModelError("歌曲对齐需要百炼音频模型。请配置百炼模型，或设置 DEEPVOCAL_AUDIO_MODEL_API_KEY。")
    return Model("dashscope", os.getenv("DEEPVOCAL_AUDIO_MODEL", DEFAULT_AUDIO_MODEL), key)


def audio_json(model, schema, content):
    payload = {"messages": [{"role": "user", "content": content}], "modalities": ["text"], "stream": False,
               "reasoning_effort": "none", "response_format": {"type": "json_object"}, "max_tokens": 1200}
    reply = model.call("/chat/completions", payload, timeout=100)
    try:
        choice = reply["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete audio model response")
        value = json.loads(choice["message"]["content"])
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and (
                schema is not Candidates or "candidates" in value[0] or "match" in value[0]):
            value = value[0]
        if schema is Candidates:
            if isinstance(value, dict) and any(key in value for key in CANDIDATE_INDEX_KEYS):
                value = {"candidates": [value], "match": value.get("match")}
            elif isinstance(value, list):
                value = {"candidates": value}
        return schema.model_validate(value)
    except (IndexError, KeyError, TypeError, ValueError):
        raise ModelError("音频模型没有返回完整的结构化结果，请重试。") from None


def text(value):
    return {"type": "text", "text": value}


def audio_item(samples, rate, target=16000):
    if rate > target:
        offsets = np.arange(-30, 31)
        cutoff = 0.45 * target / rate
        kernel = np.sinc(2 * cutoff * offsets) * np.hamming(len(offsets))
        samples = np.convolve(samples, kernel / kernel.sum(), mode="same")
    positions = np.arange(round(len(samples) * target / rate)) * rate / target
    converted = np.interp(positions, np.arange(len(samples)), samples)
    encoded = b64encode(audio.to_wave(converted, target)).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{encoded}", "format": "wav"}}


def read_media(store, key):
    return (store.media / f"{key}.wav").read_bytes()


def recording_audio(store, spec):
    samples, rate = audio.read_wave(read_media(store, spec["audio_id"]))
    offset = round(spec["offset_seconds"] * rate)
    if len(samples) - offset < rate / 2:
        raise ValueError("录音起点后需有至少 0.5 秒的演唱音频。")
    return samples[offset:], rate


def cut_wave(data, start, end):
    """Copy original PCM samples at the selected timestamps."""
    out = BytesIO()
    with wave.open(BytesIO(data), "rb") as src:
        rate = src.getframerate()
        left, right = round(start * rate), round(end * rate)
        if not 0 <= left < right <= src.getnframes():
            raise ValueError("参考片段的起止时间超出音频范围。")
        src.setpos(left)
        with wave.open(out, "wb") as target:
            target.setnchannels(src.getnchannels())
            target.setsampwidth(src.getsampwidth())
            target.setframerate(rate)
            target.writeframes(src.readframes(right - left))
    return out.getvalue()


def windows(duration, query_seconds):
    start = 0
    width = max(WINDOW_SECONDS, WINDOW_STEP + ceil(query_seconds * 1.25))
    while start < duration:
        end = min(start + width, duration)
        yield start, end
        if end == duration:
            break
        start += WINDOW_STEP


def window_batches(bounds, budget):
    batch = []
    total = 0
    for start, end in bounds:
        if batch and (len(batch) == WINDOWS_PER_REQUEST or total + end - start > budget):
            yield batch
            batch, total = [], 0
        batch.append((start, end))
        total += end - start
    if batch:
        yield batch


def absolute_candidates(response, bounds, duration):
    for entry in response.candidates:
        if entry.index > len(bounds):
            continue
        start, end = bounds[entry.index - 1]
        length = entry.end_seconds - entry.start_seconds
        if (0.5 <= length <= min(90, max(2, duration * 2)) and length >= duration * 0.32
                and entry.end_seconds <= end - start + 0.02):
            yield (round(start + entry.start_seconds, 3), round(min(end, start + entry.end_seconds), 3))


def distinct(intervals):
    kept = []
    for start, end in intervals:
        if all(min(end, right) - max(start, left) < 0.6 * min(end - start, right - left) for left, right in kept):
            kept.append((start, end))
    return kept


OUTPUT_FORMAT = '只返回 JSON {"candidates":[{"index":1,"start_seconds":1.2,"end_seconds":5.6}]}。时间相对于该候选音频起点。无可靠证据时返回空列表。'
MATCH_OUTPUT_FORMAT = ('只返回 JSON {"match":"matched|mismatch|uncertain",'
                       '"candidates":[{"index":1,"start_seconds":1.2,"end_seconds":5.6}]}。'
                       '时间相对于该候选音频起点。只有 matched 才能保留已确认的区间。')


def locate_with_model(model, query, reference, rate, query_seconds, *, verify_attachment=False):
    first_pass = []
    can_rule_out_match = True
    instructions = MATCH_INSTRUCTIONS if verify_attachment else ""
    output_format = MATCH_OUTPUT_FORMAT if verify_attachment else OUTPUT_FORMAT
    for batch in window_batches(windows(len(reference) / rate, query_seconds), 240):
        content = [text("根据两段实际音频匹配用户演唱与参考主唱的唱词和旋律。忽略伴奏、音色及整体移调。"
                        "找出完整覆盖用户有效演唱的片段。同一句出现多次时保留各处。禁止根据对歌曲的记忆猜时间。"
                        + instructions),
                   text("用户演唱"), query]
        for number, (start, end) in enumerate(batch, 1):
            content.extend([text(f"候选 {number}，本段长度 {end-start:.2f} 秒"),
                            audio_item(reference[round(start*rate):round(end*rate)], rate, 8000)])
        batch_result = audio_json(model, Candidates, [*content, text(output_format)])
        if verify_attachment and batch_result.match == "mismatch":
            can_rule_out_match &= not batch_result.candidates
            continue
        found = list(absolute_candidates(batch_result, batch, query_seconds))
        first_pass.extend(found)
        can_rule_out_match &= bool(found) and batch_result.match == "matched"
    proposals = distinct(first_pass)
    if not proposals:
        return [], verify_attachment and can_rule_out_match
    can_rule_out_match &= len(proposals) <= 8

    verify_bounds = [(max(0, start - 3), min(len(reference) / rate, end + 3)) for start, end in proposals[:8]]
    verified = []
    for batch in window_batches(verify_bounds, 145):
        content = [text("复听用户演唱和以下原曲局部。只保留能完整对应用户录音的片段，校准真正起止时间。"
                        "首尾录入的静音不计。同一句反复出现且无法区分时保留所有完整候选，禁止随意挑一遍。"
                        + instructions),
                   text("用户演唱"), query]
        for number, (start, end) in enumerate(batch, 1):
            content.extend([text(f"候选 {number}，本段长度 {end-start:.2f} 秒"),
                            audio_item(reference[round(start*rate):round(end*rate)], rate)])
        reply = audio_json(model, Candidates, [*content, text(output_format)])
        can_rule_out_match &= reply.match == "mismatch" and not reply.candidates
        if verify_attachment and reply.match != "matched":
            continue
        verified.extend(absolute_candidates(reply, batch, query_seconds))
    return distinct(verified), verify_attachment and can_rule_out_match


def alignment_report(spec, references, candidates, method, *, issue="alignment_uncertain"):
    result = report_metrics(issue) | {"recording_audio_id": spec["audio_id"],
               "offset_seconds": spec["offset_seconds"], "alignment_method": method, "candidates": candidates}
    if len(references) == 1:
        result["reference"] = references[0]
    return result


def timed_characters(model, waveform, rate):
    transcriber = Model(model.provider, "qwen-audio-3.1-asr-flash", model.key)
    base = os.getenv("DEEPVOCAL_AUDIO_ASR_BASE") or DASHSCOPE_RERANK_BASE
    reply = transcriber.call("/services/aigc/multimodal-generation/generation", {
        "input": {"messages": [{"role": "user", "content": [audio_item(waveform, rate)]}]},
        "parameters": {"format": "wav", "sample_rate": 16000},
    }, base=base, timeout=45)
    output = reply.get("output", {})
    if not isinstance(output, dict):
        return []
    nested = output.get("output", {})
    sentence = output.get("sentence") or (nested.get("sentence") if isinstance(nested, dict) else None)
    words = sentence.get("words", []) if isinstance(sentence, dict) else []
    if not isinstance(words, list):
        return []
    chars = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            continue
        letters = [char.casefold() for char in word["text"] if char.isalnum()]
        left, right = word.get("begin_time"), word.get("end_time")
        if not letters or not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or not 0 <= left < right:
            continue
        step = (right - left) / len(letters)
        chars.extend((char, left + step*i, left + step*(i+1)) for i, char in enumerate(letters))
    return chars


def edit_distance(query, reference):
    row = list(range(len(reference) + 1))
    for i, char in enumerate(query, 1):
        next_row = [i]
        for j, target in enumerate(reference, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j-1] + (char != target)))
        row = next_row
    return row[-1]


def refine_with_lyrics(store, model, recorded, rate, candidate):
    """Check the precise boundaries of a uniquely localized sung phrase."""
    try:
        query = timed_characters(model, recorded, rate)
        if not 6 <= len(query) <= 48:
            return candidate
        original, reference_rate = audio.read_wave(read_media(store, candidate["audio_id"]), audio.MAX_REFERENCE_SECONDS)
        left = max(0, round((candidate["start_seconds"] - 4) * reference_rate))
        right = min(len(original), round((candidate["end_seconds"] + 4) * reference_rate))
        reference = timed_characters(model, original[left:right], reference_rate)
    except ModelError:
        return candidate
    if len(reference) < 4:
        return None
    phrase = "".join(char for char, _, _ in query)
    available = "".join(char for char, _, _ in reference)
    max_errors = max(2, int(len(query) * 0.30))
    duration = (query[-1][2] - query[0][1]) / 1000
    matched = []
    for i in range(len(reference)):
        for j in range(i + max(4, len(phrase) - max_errors) - 1,
                       min(len(reference), i + len(phrase) + max_errors)):
            start, end = (left / reference_rate + reference[k][side] / 1000 for k, side in ((i, 1), (j, 2)))
            if not max(0.5, 0.5 * duration) <= end - start <= min(90, 1.8 * duration):
                continue
            if end > candidate["duration_seconds"] + 0.001:
                continue
            errors = edit_distance(phrase, available[i:j + 1])
            if errors <= max_errors:
                displacement = abs(start-candidate["start_seconds"]) + abs(end-candidate["end_seconds"])
                matched.append((errors, abs(j-i+1-len(phrase)), displacement, start, end))
    if not matched:
        return None
    _, _, _, start, end = min(matched)
    return {**candidate, "start_seconds": round(start, 3),
            "end_seconds": min(round(end, 3), candidate["duration_seconds"]), "boundary_source": "word_timestamps"}


def save_reference_clip(store, selected):
    start, end = selected["start_seconds"], selected["end_seconds"]
    clipped = cut_wave(read_media(store, selected["audio_id"]), start, end)
    clip_id = store.put_audio(uuid4().hex, selected["title"], "aligned_reference", clipped)
    return {**selected, "clip_audio_id": clip_id, "clip_audio_url": f"/api/audio/{clip_id}"}


def align_audio(store, spec, references, controller):
    model = audio_model(controller)
    recorded, rate = recording_audio(store, spec)
    item = audio_item(recorded, rate)
    candidates = []
    for reference in references:
        data = read_media(store, reference["audio_id"])
        original, original_rate = audio.read_wave(data, audio.MAX_REFERENCE_SECONDS)
        duration = len(original) / original_rate
        reference["duration_seconds"] = duration
        intervals, mismatch = locate_with_model(model, item, original, original_rate, len(recorded) / rate,
                                                verify_attachment=bool(spec.get("reference_audio_id")))
        if mismatch:
            return None, alignment_report(spec, references, [], "llm", issue="reference_mismatch")
        for start, end in intervals:
            candidates.append({**reference, "start_seconds": start, "end_seconds": min(end, duration)})
    if len(candidates) == 1:
        verified = refine_with_lyrics(store, model, recorded, rate, candidates[0])
        if verified:
            return save_reference_clip(store, verified), None
    return None, alignment_report(spec, references, candidates, "llm")


def selected_reference(store, spec, reference, controller):
    data = read_media(store, reference["audio_id"])
    with wave.open(BytesIO(data), "rb") as src:
        duration = src.getnframes() / src.getframerate()
    start, end = spec["reference_start_seconds"], spec["reference_end_seconds"]
    if end > duration + 0.001 or end - start < 0.5:
        raise ValueError("参考片段的起止时间超出音频范围。")
    end = min(end, duration)
    excerpt, excerpt_rate = audio.read_wave(cut_wave(data, start, end), audio.MAX_REFERENCE_SECONDS)
    recorded, rate = recording_audio(store, spec)
    verdict = audio_json(audio_model(controller), MatchCheck, [
        text(MATCH_INSTRUCTIONS + '只输出 JSON {"match":"matched|mismatch|uncertain"}。'),
        text("用户演唱"), audio_item(recorded, rate), text("选中的参考片段"), audio_item(excerpt, excerpt_rate),
    ])
    if verdict.match != "matched":
        # A selected range can miss the right verse even when the full song is correct.
        issue = "reference_segment_mismatch" if verdict.match == "mismatch" else "alignment_uncertain"
        return None, alignment_report(spec, [reference], [], "manual", issue=issue)
    return save_reference_clip(store, {**reference, "start_seconds": start, "end_seconds": end,
                                       "duration_seconds": duration}), None


def assess_aligned_audio(store, spec, selected, controller):
    model = audio_model(controller)
    recorded, rate = recording_audio(store, spec)
    start, end = selected["start_seconds"], selected["end_seconds"]
    source, source_rate = audio.read_wave(read_media(store, selected["clip_audio_id"]), audio.MAX_REFERENCE_SECONDS)
    instruction = ("你是声乐导师。对比实际演唱 A 与已经选定的原曲片段 B，忽略 A 首尾静音和 B 伴奏。"
                   "观察相对音准、节奏和稳定性。整体换调与局部唱偏分开描述。给出一条能执行的练习。"
                   "只写能听到的定性观察，不捏造分数、音分、毫秒、音符计数或身体诊断。"
                   "若两段实际不对应或人声无法判断，assessable=false，summary 给出具体原因。"
                   '只输出 JSON {"assessable":true,"focus":"general|pitch_flat|pitch_sharp|timing|stability",'
                   '"summary":"简短结论","intonation":"音准","rhythm":"节奏","stability":"稳定性","exercise":"练习"}。')
    feedback = audio_json(model, Feedback, [text(instruction), text("A 用户演唱"), audio_item(recorded, rate),
                                           text("B 对齐后的原曲片段"), audio_item(source, source_rate)])
    markers = {**selected, "selection": "manual" if spec.get("reference_start_seconds") is not None else "model",
               "transpose": spec["transpose"], "segment_start_seconds": start, "segment_end_seconds": end,
               "segment_id": uuid5(NAMESPACE_URL, f"deepvocal:{selected['audio_id']}:{round(start*1000)}:{round(end*1000)}").hex}
    return report_metrics(feedback.focus if feedback.assessable else "insufficient_audio") | {
        "assessment_method": "audio_model", "alignment_method": markers["selection"] if markers["selection"] == "manual" else "llm",
        "audio_model": model.name,
        "recording_audio_id": spec["audio_id"], "offset_seconds": spec["offset_seconds"],
        "recording_end_seconds": round(spec["offset_seconds"] + len(recorded) / rate, 3),
        "reference": markers, "audio_feedback": feedback.model_dump(),
    }
