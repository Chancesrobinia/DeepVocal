"""Optional RMVPE and constrained subsequence DTW candidate locator."""

from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

import numpy as np

from . import audio
from .audio_alignment import alignment_report, distinct, read_media, recording_audio


HOP = 0.08
MIN_COVERAGE = 0.82
MODEL_LOCK = Lock()


@lru_cache(maxsize=2)
def load_rmvpe(path):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ValueError("备用 DTW 需要安装 onnxruntime。") from None
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    return ort.InferenceSession(path, settings, providers=["CPUExecutionProvider"])


def pitch_frames(model, samples, rate):
    if rate > 16000:
        offsets = np.arange(-30, 31)
        kernel = np.sinc(0.9 * 16000 / rate * offsets) * np.hamming(len(offsets))
        samples = np.convolve(samples, kernel / kernel.sum(), mode="same")
    count = round(len(samples) * 16000 / rate)
    signal = np.interp(np.arange(count) * rate / 16000, np.arange(len(samples)), samples).astype(np.float32)
    stride = round(16000 * HOP)
    pitch = np.full(count // stride, np.nan)
    # Keep context at each internal chunk boundary while retaining original frame timestamps.
    for first in range(0, len(pitch), 125):
        last = min(first + 125, len(pitch))
        left = max(0, first - 12)
        data = signal[left * stride:min(count, (last + 12) * stride)]
        output = model.run(None, {model.get_inputs()[0].name: data[None, :]})[0][0]
        vectors = output[np.arange(first - left, last - left) * 8]
        peaks = vectors.argmax(axis=1)
        bins = np.clip(peaks[:, None] + np.arange(-4, 5), 0, 359)
        weights = np.take_along_axis(vectors, bins, axis=1)
        notes = (bins * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9) * 0.2
        pitch[first:last] = np.where(vectors.max(axis=1) >= 0.22, notes, np.nan)
    return pitch


def backtrack(back, n, end):
    i, j = n, end
    pairs = []
    while i and j:
        pairs.append((i - 1, j - 1))
        move = back[i, j]
        if move == 1:
            i, j = i - 1, j - 1
        elif move == 2:
            pairs.append((i - 1, j - 2))
            i, j = i - 1, j - 2
        else:
            pairs.append((i - 2, j - 1))
            i, j = i - 2, j - 1
    return i, j, np.asarray(pairs).T


def subsequence_candidates(reference, recording):
    active = np.flatnonzero(np.isfinite(recording))
    if len(active) < 8 or len(reference) < 8:
        return []
    query = recording[active[0]:active[-1] + 1]
    n, m = len(query), len(reference)
    delta = query[:, None] - reference[None, :]
    both = np.isfinite(delta)
    at_least_one = np.isfinite(query[:, None]) | np.isfinite(reference[None, :])
    penalized = np.where(at_least_one, 1.4, 0.05)
    found = []
    for shift in range(-12, 13):
        costs = np.where(both, np.minimum(np.abs(delta - shift), 3.5), penalized)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, :] = 0
        back = np.zeros(dp.shape, dtype=np.uint8)
        for i, row in enumerate(costs, 1):
            best = dp[i - 1, :m] + 2 * row
            move = np.ones(m, dtype=np.uint8)
            right = dp[i - 1, :m - 1] + 1.5 * (row[1:] + row[:-1]) + 0.32
            use = right < best[1:]
            best[1:][use] = right[use]
            move[1:][use] = 2
            if i >= 2:
                down = dp[i - 2, :m] + 1.5 * (row + costs[i - 2]) + 0.32
                use = down < best
                best[use] = down[use]
                move[use] = 3
            dp[i, 1:] = best
            back[i, 1:] = move
        ends = dp[n].copy()
        separation = max(4, round(n * 0.26))
        for _ in range(18):
            end = int(ends.argmin())
            if not np.isfinite(ends[end]):
                break
            ends[max(0, end - separation):end + separation + 1] = np.inf
            i, start, path = backtrack(back, n, end)
            if i or not 0.70 <= (end - start) / n <= 1.45:
                continue
            x, y = path
            if len(np.unique(x[both[x, y]])) / len(active) >= MIN_COVERAGE:
                found.append((float(dp[n, end] / (n + end - start)), start * HOP, end * HOP))
    intervals = [(round(start, 3), round(end, 3)) for _, start, end in sorted(found)]
    return distinct(intervals)[:3]


def align_dtw(store, spec, references):
    path = Path(os.getenv("DEEPVOCAL_RMVPE_ONNX_PATH") or store.root / "models" / "rmvpe.onnx")
    if not path.is_file():
        raise ValueError("备用 DTW 需要 RMVPE ONNX 模型，请设置 DEEPVOCAL_RMVPE_ONNX_PATH。")
    with MODEL_LOCK:
        model = load_rmvpe(str(path.resolve()))
    recording = pitch_frames(model, *recording_audio(store, spec))
    candidates = []
    for reference in references:
        samples, rate = audio.read_wave(read_media(store, reference["audio_id"]), audio.MAX_REFERENCE_SECONDS)
        reference["duration_seconds"] = len(samples) / rate
        intervals = subsequence_candidates(pitch_frames(model, samples, rate), recording)
        candidates.extend({**reference, "start_seconds": start, "end_seconds": end} for start, end in intervals)
    return alignment_report(spec, references, candidates, "dtw")
