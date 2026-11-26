"""Convert a vocal clip with Seed-VC's Gradio singing endpoint."""

import json
import os
import re
from time import monotonic
from uuid import uuid4

import httpx
import numpy as np

from . import audio


DEFAULT_SEED_VC_URL = "https://plachta-seed-vc.hf.space"
LOCAL_SEED_VC_URL = "http://127.0.0.1:7860"
TIMBRE_SECONDS = 25
SOURCE_SECONDS = 60
SONG_CHUNK_SECONDS = 20
SONG_OVERLAP_SECONDS = 0.25
RESULT_TIMEOUT = 180
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
SEED_INPUTS = {
    "diffusion_steps": 30, "length_adjust": 1.0, "inference_cfg_rate": 0.7,
    "f0_condition": True, "auto_f0_adjust": False,
}


class VoiceConversionError(ValueError):
    """A conversion failure that can be displayed to the caller."""


def service_url(*, local_only=False):
    try:
        default = LOCAL_SEED_VC_URL if local_only else DEFAULT_SEED_VC_URL
        url = httpx.URL(os.getenv("DEEPVOCAL_SEED_VC_URL", default).strip().rstrip("/"))
        if (not url.host or url.userinfo or url.query or url.fragment
                or (url.scheme != "https" and not (url.scheme == "http" and url.host in LOOPBACK_HOSTS))):
            raise VoiceConversionError("请配置有效的 Seed-VC Gradio 服务地址。")
    except httpx.InvalidURL:
        raise VoiceConversionError("请配置有效的 Seed-VC Gradio 服务地址。") from None
    if local_only and url.host not in LOOPBACK_HOSTS:
        raise VoiceConversionError("歌曲翻唱需要本机 Seed-VC 服务。请用本地推理配置启动 DeepVocal。")
    return url


def _singing_endpoint(client):
    configuration = client.get("config")
    configuration.raise_for_status()
    prefix = configuration.json().get("api_prefix") or "/gradio_api"
    if not re.fullmatch(r"/[a-zA-Z0-9_-]+", prefix):
        raise VoiceConversionError("Seed-VC 返回的 Gradio 接口地址无效。")
    prefix = prefix.strip("/")
    response = client.get(f"{prefix}/info")
    response.raise_for_status()
    dependencies = {item["api_name"]: item["id"] for item in configuration.json()["dependencies"]
                    if item.get("backend_fn") and item.get("api_name") and type(item.get("id")) is int}
    candidates = []
    for route, schema in response.json()["named_endpoints"].items():
        name = route.removeprefix("/")
        if name not in dependencies:
            continue
        parameters = [item["parameter_name"] for item in schema["parameters"]]
        source_name = next((key for key in ("source_audio_path", "source") if key in parameters), None)
        target_name = next((key for key in ("target_audio_path", "target") if key in parameters), None)
        expected = {source_name, target_name, "pitch_shift", *SEED_INPUTS}
        if "f0_condition" not in parameters:
            expected.remove("f0_condition")
        if not (source_name and target_name and set(parameters) == expected):
            continue
        full_audio = next((i for i, item in enumerate(schema["returns"])
                           if (item.get("label") or "").startswith("Full Output Audio")), None)
        if full_audio is not None:
            candidates.append(("f0_condition" in parameters, dependencies[name], source_name,
                               target_name, parameters, full_audio))
    if not candidates:
        raise VoiceConversionError("Seed-VC 服务没有公开兼容的歌声转换接口。")
    _, index, source_name, target_name, parameters, full_audio = max(candidates, key=lambda item: item[0])
    return prefix, index, source_name, target_name, parameters, full_audio


def _prediction_failure(message):
    text = message.casefold() if isinstance(message, str) else ""
    if "quota" in text or "zerogpu runs limit" in text:
        return VoiceConversionError("Seed-VC 的 Hugging Face GPU 配额已用完。请配置可用的 Hugging Face Token 或更换推理服务。")
    return VoiceConversionError("Seed-VC 未能完成歌声转换，请检查音频或推理服务。")


def _await_prediction(client, prefix, session, event_id):
    deadline = monotonic() + RESULT_TIMEOUT
    with client.stream("GET", f"{prefix}/queue/data", params={"session_hash": session}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if monotonic() >= deadline:
                raise VoiceConversionError("Seed-VC 任务等待超时，请稍后重试。")
            if not line.startswith("data:"):
                continue
            message = json.loads(line[5:])
            if message.get("event_id") not in (None, event_id):
                continue
            if message.get("msg") == "process_completed":
                output = message.get("output") or {}
                if not message.get("success"):
                    raise _prediction_failure(output.get("error"))
                return output["data"]
            if message.get("msg") in ("unexpected_error", "server_stopped", "queue_full"):
                raise _prediction_failure(message.get("message"))
    raise VoiceConversionError("Seed-VC 连接已结束，未返回完整音频。")


def _download_wav(client, prefix, result, index):
    location = result[index]["url"]
    url = client.base_url.join(location)
    expected_path = client.base_url.path + f"{prefix}/file="
    if (url.scheme, url.host, url.port) != (client.base_url.scheme, client.base_url.host, client.base_url.port) or (
            url.userinfo or not url.path.startswith(expected_path) or result[index].get("is_stream")):
        raise VoiceConversionError("Seed-VC 返回的完整音频地址无效。")
    data = bytearray()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if len(data) + len(chunk) > audio.MAX_BYTES:
                raise VoiceConversionError("Seed-VC 返回的 WAV 超过大小限制。")
            data.extend(chunk)
    return bytes(data)


def _seed_vc(source, timbre, semitone_shift, *, local_only=False):
    url = service_url(local_only=local_only)
    token = (os.getenv("DEEPVOCAL_SEED_VC_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token and url.host.endswith(".hf.space") else {}
    try:
        with httpx.Client(base_url=str(url), headers=headers, timeout=httpx.Timeout(45, connect=10),
                          trust_env=url.host not in LOOPBACK_HOSTS, follow_redirects=False) as client:
            prefix, index, source_name, target_name, parameters, full_audio = _singing_endpoint(client)
            response = client.post(f"{prefix}/upload", files=[
                ("files", ("source-vocal.wav", source, "audio/wav")),
                ("files", ("timbre.wav", timbre, "audio/wav")),
            ])
            response.raise_for_status()
            paths = response.json()
            if not isinstance(paths, list) or len(paths) != 2 or not all(isinstance(path, str) and path for path in paths):
                raise VoiceConversionError("Seed-VC 未确认收到两份音频。")
            files = {key: {"path": path, "meta": {"_type": "gradio.FileData"}}
                     for key, path in zip((source_name, target_name), paths, strict=True)}
            values = {**SEED_INPUTS, **files, "pitch_shift": semitone_shift}
            session = uuid4().hex
            queued = client.post(f"{prefix}/queue/join", json={
                "session_hash": session, "fn_index": index, "data": [values[name] for name in parameters],
            })
            queued.raise_for_status()
            output = _await_prediction(client, prefix, session, queued.json()["event_id"])
            return _download_wav(client, prefix, output, full_audio)
    except VoiceConversionError:
        raise
    except httpx.TimeoutException:
        raise VoiceConversionError("Seed-VC 请求超时，请稍后重试。") from None
    except httpx.HTTPStatusError as error:
        if error.response.status_code in (401, 403):
            raise VoiceConversionError("Seed-VC 拒绝访问，请检查 Hugging Face Token 和 Space 权限。") from None
        if error.response.status_code == 429:
            raise VoiceConversionError("Seed-VC 请求受限，请检查 Hugging Face GPU 配额或稍后重试。") from None
        raise VoiceConversionError("Seed-VC 服务暂时无法处理这次转换。") from None
    except httpx.HTTPError:
        if local_only:
            raise VoiceConversionError("无法连接本机 Seed-VC，请确认本地推理服务正在运行。") from None
        raise VoiceConversionError("无法连接 Seed-VC 服务，请检查网络或服务地址。") from None
    except (KeyError, IndexError, TypeError, AttributeError, ValueError, httpx.InvalidURL):
        raise VoiceConversionError("Seed-VC 返回了无法识别的响应。") from None


def convert_singing_voice(source_vocal: bytes, timbre: bytes, *, semitone_shift: int = 0, local_only=False) -> bytes:
    """Convert source phrasing and pitch to the target timbre from a different WAV."""
    if type(semitone_shift) is not int or not -12 <= semitone_shift <= 12:
        raise VoiceConversionError("移调需为 -12 至 12 个整数半音。")
    try:
        source_samples, source_rate = audio.read_wave(source_vocal, SOURCE_SECONDS)
    except ValueError:
        raise VoiceConversionError("请提供 60 秒以内、能正常播放的参考主唱片段。") from None
    try:
        timbre_samples, timbre_rate = audio.read_wave(timbre)
    except ValueError:
        raise VoiceConversionError("请提供能正常播放的本人录音。") from None
    timbre_samples = timbre_samples[:timbre_rate * TIMBRE_SECONDS]
    if not source_samples.any() or not timbre_samples.any():
        raise VoiceConversionError("参考主唱和本人音色录音都需要包含可听见的声音。")
    wav = _seed_vc(audio.to_wave(source_samples, source_rate), audio.to_wave(timbre_samples, timbre_rate),
                   semitone_shift, local_only=local_only)
    try:
        samples, rate = audio.read_wave(wav, SOURCE_SECONDS + 1)
    except ValueError:
        raise VoiceConversionError("Seed-VC 未返回可播放的 WAV。") from None
    original_duration = len(source_samples) / source_rate
    if not samples.any() or abs(len(samples) / rate - original_duration) > max(0.5, original_duration * 0.05):
        raise VoiceConversionError("Seed-VC 返回了静音或时长异常的音频。")
    return wav


def convert_song(source_vocal, timbre, *, semitone_shift, start_seconds=None, end_seconds=None, progress=None):
    """Render a reference with overlapping windows, preserving its timing and pauses."""
    source, rate = audio.read_wave(source_vocal, audio.MAX_REFERENCE_SECONDS)
    if start_seconds is not None:
        if end_seconds > len(source) / rate:
            raise VoiceConversionError("参考片段超出了参考歌曲的时长。")
        source = source[round(start_seconds * rate):round(end_seconds * rate)]
    timbre_samples, timbre_rate = audio.read_wave(timbre)
    if not source.any() or not timbre_samples[:TIMBRE_SECONDS * timbre_rate].any():
        raise VoiceConversionError("参考主唱和本人音色录音都需要包含可听见的声音。")
    window = rate * SONG_CHUNK_SECONDS
    if len(source) <= window:
        return convert_singing_voice(audio.to_wave(source, rate), timbre, semitone_shift=semitone_shift, local_only=True)

    overlap = round(rate * SONG_OVERLAP_SECONDS)
    # Start each new window in the preceding one. Never create a window consisting only of overlap.
    starts = range(0, len(source) - overlap, window - overlap)
    output = None
    output_rate = None
    blended_until = 0
    for index, start in enumerate(starts, 1):
        end = min(start + window, len(source))
        clip = source[start:end]
        if clip.any():
            wav = convert_singing_voice(audio.to_wave(clip, rate), timbre,
                                        semitone_shift=semitone_shift, local_only=True)
            generated, generated_rate = audio.read_wave(wav, SONG_CHUNK_SECONDS + 1)
            if output_rate is None:
                output_rate = generated_rate
                output = np.zeros(round(len(source) * output_rate / rate), dtype=np.float32)
                blended_until = round(start * output_rate / rate)
            if generated_rate != output_rate:
                raise VoiceConversionError("Seed-VC 返回的歌曲分段采样率不一致。")
        else:
            generated = np.empty(0, dtype=np.float32)
        if output is not None:
            left, right = round(start * output_rate / rate), round(end * output_rate / rate)
            samples = np.zeros(right - left, dtype=np.float32)
            count = min(len(generated), len(samples))
            samples[:count] = generated[:count]
            fade_length = max(0, min(blended_until, right) - left)
            fade = np.linspace(0, 1, fade_length, dtype=np.float32)
            samples[:fade_length] = output[left:left + fade_length] * (1 - fade) + samples[:fade_length] * fade
            output[left:right] = samples
            blended_until = right
        if progress is not None:
            progress(index, len(starts))
    return audio.to_wave(output, output_rate)
