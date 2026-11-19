"""Local HTTP adapter for MoonInTheRiver/DiffSinger's OpenCpop cascade."""
import argparse
from array import array
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import wave


def score(payload):
    lyrics = "".join(payload["lyrics"].split())
    notes = payload["midi_notes"]
    bpm = payload["bpm"]
    beats = payload["beats_per_note"]
    if not re.fullmatch(r"[\u4e00-\u9fff]{1,32}", lyrics) or len(notes) != len(lyrics):
        raise ValueError("Mandarin lyrics must contain one Han character per note.")
    if any(type(midi) is not int or not 36 <= midi <= 84 for midi in notes):
        raise ValueError("MIDI notes must be between 36 and 84.")
    if not 50 <= bpm <= 160 or beats != 1 or payload.get("voice") not in ("default", "opencpop"):
        raise ValueError("Unsupported tempo, note length, or voice.")
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return {
        "text": lyrics, "input_type": "word",
        "notes": " | ".join(f"{names[n % 12]}{n // 12 - 1}" for n in notes),
        "notes_duration": " | ".join([f"{60 / bpm:.6f}"] * len(notes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from inference.svs.ds_cascade import DiffSingerCascadeInfer
    from utils.hparams import hparams, set_hparams

    set_hparams(config=args.config, exp_name=args.exp_name, print_hparams=False)
    model = DiffSingerCascadeInfer(hparams)
    rate = hparams["audio_sample_rate"]
    token = os.getenv("DEEPVOCAL_SINGING_TOKEN", "").encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/sing":
                self.send_error(404)
                return
            if token and not compare_digest(self.headers.get("Authorization", "").encode(), b"Bearer " + token):
                self.send_error(401)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError
                self.connection.settimeout(10)
                inputs = score(json.loads(self.rfile.read(size)))
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(400, "Provide one Mandarin character per note and a supported tempo and voice.")
                return
            try:
                samples = model.infer_once(inputs)
                pcm = array("h", (round(max(-1, min(1, float(s))) * 32767) for s in samples))
                if sys.byteorder != "little":
                    pcm.byteswap()
                buf = io.BytesIO()
                with wave.open(buf, "wb") as output:
                    output.setparams((1, 2, rate, 0, "NONE", ""))
                    output.writeframes(pcm.tobytes())
                data = buf.getvalue()
            except Exception:
                self.send_error(422, "DiffSinger could not synthesize this score.")
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
