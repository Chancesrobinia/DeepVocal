import io
import os
import time
import wave
from itertools import chain

import av
import httpx
import numpy as np


MELODIES = {
    "warmup": {"title": "八音上行下行练习", "notes": [60, 62, 64, 65, 67, 65, 64, 60]},
    "triad": {"title": "五音三和弦练习", "notes": [60, 64, 67, 64, 60]},
}
MAX_BYTES = 12 * 1024 * 1024
MAX_REFERENCE_SECONDS = 480
MAX_REFERENCE_BYTES = 96 * 1024 * 1024
SCORING_VERSION = 1
UNUSABLE_ISSUES = {"insufficient_audio", "alignment_uncertain", "reference_unusable", "reference_not_found",
                   "song_title_required", "reference_mismatch", "reference_segment_mismatch"}


def read_wave(data, max_seconds=60):
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            rate = reader.getframerate()
            frames = reader.getnframes()
            if (reader.getsampwidth() != 2 or channels not in (1, 2) or not 8000 <= rate <= 48000
                    or not 0.25 <= frames / rate <= max_seconds):
                raise ValueError(f"音频需为 16 位 PCM WAV，单声道或立体声，8–48 kHz，0.25–{max_seconds} 秒。")
            samples = np.frombuffer(reader.readframes(frames), dtype="<i2")
            if len(samples) != frames * channels:
                raise ValueError("WAV 数据不完整。")
    except (wave.Error, EOFError):
        raise ValueError("请上传有效的 16 位 PCM WAV 文件。") from None
    return samples.reshape((-1, channels)).mean(axis=1, dtype=np.float32) / 32768, rate


def decode_m4a(data, max_seconds):
    pcm = bytearray()
    duration_error = f"M4A 音频时长须在 0.25–{max_seconds} 秒之间。"
    try:
        with av.open(io.BytesIO(data), format="mov", options={"enable_drefs": "0"}) as source:
            if not source.streams.audio:
                raise ValueError("M4A 文件中没有可用的音轨。")
            stream = source.streams.audio[0]
            source_rate = stream.rate or 44100
            rate = min(48000, max(8000, source_rate))
            duration = stream.duration * stream.time_base if stream.duration and stream.time_base else None
            if duration is not None and duration > max_seconds:
                raise ValueError(duration_error)
            frames = round(duration * rate) if duration is not None else 0
            # AAC may decode a full final block even when the container marks a shorter duration.
            padding = round(2048 * rate / source_rate) if frames and stream.codec_context.name == "aac" else 0
            limit = 2 * (max_seconds * rate + padding)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            for frame in chain(source.decode(stream), (None,)):
                for converted in resampler.resample(frame):
                    if len(pcm) + converted.samples * 2 > limit:
                        raise ValueError(duration_error)
                    pcm.extend(converted.to_ndarray().astype("<i2", copy=False).tobytes())
    except av.error.FFmpegError:
        raise ValueError("无法读取 M4A 音频，请检查文件是否完整。") from None
    if 0 < len(pcm) // 2 - frames <= padding:
        del pcm[frames * 2:]
    if not 0.25 <= len(pcm) / (2 * rate) <= max_seconds:
        raise ValueError(duration_error)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", ""))
        writer.writeframes(pcm)
    return output.getvalue()


def read_upload(data, filename, max_seconds=60):
    if not data.startswith(b"RIFF") and ((filename or "").lower().endswith(".m4a") or data[4:8] == b"ftyp"):
        data = decode_m4a(data, max_seconds)
    samples, rate = read_wave(data, max_seconds)
    return data, samples, rate


def to_wave(samples, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", ""))
        writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def hz(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def guide(notes, bpm, transpose):
    rate = 22050
    duration = 60 / bpm
    count = round(duration * rate)
    t = np.arange(count) / rate
    ramp = np.minimum(np.minimum(t / 0.02, (duration * 0.92 - t) / 0.05), 1).clip(0)
    parts = []
    for note in notes:
        phase = 2 * np.pi * hz(note + transpose) * t
        voice = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        parts.append(0.42 * voice * ramp)
    return to_wave(np.concatenate(parts), rate)


def pitch(frame, sr):
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 0.009:
        return None
    size = 1 << (2 * len(frame) - 1).bit_length()
    power = abs(np.fft.rfft(frame, n=size)) ** 2
    raw = np.fft.irfft(power, n=size)
    high, low = min(len(frame) // 2, sr // 65), max(2, sr // 1100)
    corr = raw[:high + 2] / raw[0]
    peaks = [i for i in range(low, high + 1) if corr[i] >= corr[i - 1] and corr[i] > corr[i + 1]]
    if not peaks:
        return None
    best = max(corr[i] for i in peaks)
    if best < 0.55:
        return None
    i = next(i for i in peaks if corr[i] >= max(0.55, best * 0.90))
    curvature = corr[i - 1] - 2 * corr[i] + corr[i + 1]
    shift = 0.5 * (corr[i - 1] - corr[i + 1]) / curvature if abs(curvature) > 1e-9 else 0
    return sr / (i + shift)


def assess(data, spec):
    samples, sr = read_wave(data)
    melody = MELODIES[spec["melody_id"]]["notes"]
    step = 60 / spec["bpm"]
    offset = spec["offset_seconds"]
    found = []
    timing = []
    all_errors = []
    frame = round(sr * 0.05)
    for index, note in enumerate(melody):
        start = offset + index * step
        target = hz(note + spec["transpose"])
        cents = []
        lo, hi = round((start + 0.18 * step) * sr), min(len(samples), round((start + 0.88 * step) * sr))
        for point in range(lo, hi - frame + 1, max(1, round(sr * 0.02))):
            f0 = pitch(samples[point:point + frame], sr)
            if f0:
                cents.append(float(1200 * np.log2(f0 / target)))
        if len(cents) < 3:
            continue
        all_errors.extend(cents)
        # Report timing only when a quiet-to-voiced transition is visible near the note.
        hop = max(1, sr // 100)
        onset = None
        left = max(0, round((start - 0.25 * step) * sr))
        right = min(len(samples) - hop, round((start + 0.35 * step) * sr))
        previous = 0 if left == 0 else float(np.sqrt(np.mean(samples[max(0, left - hop):left] ** 2)))
        for pos in range(left, right, hop):
            rms = float(np.sqrt(np.mean(samples[pos:pos + hop] ** 2)))
            if previous < 0.014 and rms >= 0.014 and np.sqrt(np.mean(samples[pos:pos + 3 * hop] ** 2)) >= 0.02:
                onset = round((pos / sr - start) * 1000)
                timing.append(onset)
                break
            previous = rms
        found.append({"note": index + 1, "target_midi": note + spec["transpose"],
                      "start_seconds": round(start, 3), "pitch_error_cents": round(float(np.median(cents)), 1),
                      "pitch_iqr_cents": round(float(np.percentile(cents, 75) - np.percentile(cents, 25)), 1),
                      "onset_error_ms": onset})
    median = round(float(np.median(all_errors)), 1) if all_errors else None
    ratio = round(float(np.mean(np.abs(all_errors) <= 50)), 3) if all_errors else None
    issue = "insufficient_audio"
    if len(found) >= (len(melody) + 1) // 2:
        typical = float(np.median([entry["pitch_error_cents"] for entry in found]))
        if typical < -35:
            issue = "pitch_flat"
        elif typical > 35:
            issue = "pitch_sharp"
        elif len(timing) >= 2 and np.median(np.abs(timing)) > 85:
            issue = "timing"
        elif np.median([entry["pitch_iqr_cents"] for entry in found]) > 55:
            issue = "stability"
        else:
            issue = "general"
    return {"melody_id": spec["melody_id"], "observed_at": time.time(), "issue": issue, "scoring_version": SCORING_VERSION,
            "matched_notes": len(found), "total_notes": len(melody), "median_pitch_error_cents": median,
            "within_50_cents_ratio": ratio, "timing_observations": len(timing), "notes": found}


def generate(spec):
    notes = MELODIES[spec["melody_id"]]["notes"]
    if not spec.get("lyrics"):
        return guide(notes, spec["bpm"], spec["transpose"]), "guide"
    endpoint = os.environ.get("DEEPVOCAL_SINGING_URL")
    if not endpoint:
        raise ValueError("歌词合唱需要配置 DEEPVOCAL_SINGING_URL。")
    token = os.environ.get("DEEPVOCAL_SINGING_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.stream("POST", endpoint, headers=headers, timeout=60, json={
            "lyrics": spec["lyrics"], "midi_notes": [n + spec["transpose"] for n in notes],
            "beats_per_note": 1, "bpm": spec["bpm"], "voice": spec["voice"],
        }) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError("歌声服务返回的 WAV 超过大小限制。")
    except httpx.HTTPError:
        raise ValueError("歌声合成服务调用失败，请检查服务配置。") from None
    read_wave(data)
    return bytes(data), "singing"
