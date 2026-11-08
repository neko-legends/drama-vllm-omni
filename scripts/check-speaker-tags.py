#!/usr/bin/env python3
"""Show, from the audio itself, whether the server honours <|speaker:N|> tags.

Sends the same sentence tagged <|speaker:0|> and <|speaker:1|>, measures the median
pitch (f0) of each take, and then splits a multi-turn two-speaker take at its
silences and measures f0 per utterance: alternating values mean alternating voices.

Needs numpy (the only non-stdlib import) and a running server.

  ./scripts/check-speaker-tags.py --out /tmp/drama3-speakers
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import wave
from urllib.parse import urlparse

import numpy as np

LINE = "Good evening, the harbour lights are coming on."
TURNS = (
    "<|speaker:0|> You said the shipment left on Tuesday morning. "
    "<|speaker:1|> It did, but the port held it for inspection. "
    "<|speaker:0|> Then why is it still standing in the yard? "
    "<|speaker:1|> Because that is exactly where the inspectors put it."
)


def speak(base, text, timeout=300):
    u = urlparse(base)
    body = json.dumps({"model": "fishaudio/drama-3", "input": text, "response_format": "wav"}).encode()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("POST", "/v1/audio/speech", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError("HTTP %s: %s" % (resp.status, data[:300]))
        return data
    finally:
        conn.close()


def load(wav_bytes, path=None):
    if path:
        with wave.open(path) as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    else:
        import io
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def f0_median(samples, rate):
    """Median f0 over voiced 46 ms frames, by normalised autocorrelation (70-800 Hz)."""
    frame, hop = 2048, 1024
    lo, hi = int(rate / 800), int(rate / 70)
    frames = [samples[at:at + frame] for at in range(0, max(0, len(samples) - frame), hop)]
    rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames]) if frames else np.array([0.0])
    # voices differ in level, so voiced means loud for this take, not loud in absolute terms
    floor = max(0.002, 0.15 * float(np.percentile(rms, 95)))
    values = []
    for block in frames:
        if float(np.sqrt(np.mean(block ** 2))) < floor:
            continue
        block = block - block.mean()
        corr = np.correlate(block, block, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window = corr[lo:hi]
        if not len(window):
            continue
        peak = int(np.argmax(window)) + lo
        if corr[peak] > 0.3:
            values.append(rate / peak)
    return (round(float(np.median(values)), 1) if values else None), len(values)


def utterances(samples, rate, gap_s=0.22, min_s=0.14):
    """Split on silence: list of (start_s, end_s) for stretches above 3% of peak amplitude."""
    frame, hop = 512, 256
    env = np.array([np.sqrt(np.mean(samples[at:at + frame] ** 2))
                    for at in range(0, max(1, len(samples) - frame), hop)])
    loud = env > max(0.02, 0.03 * env.max())
    spans, start = [], None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start * hop / rate, i * hop / rate))
            start = None
    if start is not None:
        spans.append((start * hop / rate, len(samples) / rate))
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < gap_s:
            merged[-1] = (merged[-1][0], b)
        elif b - a >= min_s:
            merged.append((a, b))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18793")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    takes = {}
    for label, text in (("speaker0", "<|speaker:0|> " + LINE),
                        ("speaker1", "<|speaker:1|> " + LINE),
                        ("speaker0_repeat", "<|speaker:0|> " + LINE),
                        ("twoturn", TURNS)):
        data = speak(args.base, text)
        path = os.path.join(args.out, "take-%s.wav" % label)
        with open(path, "wb") as fh:
            fh.write(data)
        samples, rate = load(data)
        takes[label] = {
            "wav": path,
            "bytes": len(data),
            "audio_s": round(len(samples) / rate, 3),
            "md5": hashlib.md5(data).hexdigest(),
            "f0_median_hz": f0_median(samples, rate)[0],
            "voiced_frames": f0_median(samples, rate)[1],
        }
        print("%-15s %6.2fs  f0 %s Hz  %s" % (label, takes[label]["audio_s"],
                                             takes[label]["f0_median_hz"], takes[label]["md5"]), file=sys.stderr)

    samples, rate = load(None, takes["twoturn"]["wav"])
    segs = utterances(samples, rate)
    per_turn = []
    for i, (a, b) in enumerate(segs):
        f0, n = f0_median(samples[int(a * rate):int(b * rate)], rate)
        per_turn.append({"turn": i + 1, "start_s": round(a, 2), "end_s": round(b, 2),
                         "f0_median_hz": f0, "voiced_frames": n})
        print("turn %d  %5.2f-%5.2fs  f0 %s Hz" % (i + 1, a, b, f0), file=sys.stderr)

    report = {
        "line": LINE,
        "turns": TURNS,
        "takes": takes,
        "two_speaker_turns": per_turn,
        "speaker0_and_speaker1_differ": takes["speaker0"]["md5"] != takes["speaker1"]["md5"],
        "speaker0_repeat_is_deterministic": takes["speaker0"]["md5"] == takes["speaker0_repeat"]["md5"],
    }
    with open(os.path.join(args.out, "speakers.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "turns"}, indent=2))


if __name__ == "__main__":
    main()
