#!/usr/bin/env python3
"""Measure a running Drama 3 server: whole-WAV latency and streamed first-chunk latency.

Stdlib only, no deps. It starts nothing and stops nothing; point it at a live
`POST /v1/audio/speech` endpoint (default http://127.0.0.1:18793).

  ./scripts/bench-drama3.py --out /tmp/drama3-bench --runs 3

Writes <out>/bench.json, the WAV takes, and <out>/stream/*.pcm.
Reported per request: time to first body byte (TTFB), total wall time, generated
audio seconds (from the WAV payload length, 16-bit mono 44.1 kHz), and the
real-time factor RTF = audio seconds / wall seconds.
"""
import argparse
import hashlib
import http.client
import json
import os
import statistics
import sys
import time
from urllib.parse import urlparse

WAV_HEADER = 44
RATE = 44100
BYTES_PER_SAMPLE = 2


def audio_seconds(nbytes, header=0):
    return max(0, nbytes - header) / (RATE * BYTES_PER_SAMPLE)


TEXTS = {
    "short": (
        "The train leaves at seven fifteen and we should hurry.",
        10,
    ),
    "medium": (
        "Every autumn the harbour fills with boats from the north. The crews spend the "
        "first week repairing nets and arguing about the weather. By the second week the "
        "market stalls open at dawn, selling smoked fish, bread and coffee. Crane "
        "operators say the season is short, but nobody believes them.",
        50,
    ),
    "hundred": (
        "The night shift at the water plant begins with a walk along the filter gallery, "
        "where the pumps hum in a rhythm the older operators can read like a clock. If the "
        "third pump sounds flat, they say, the intake screen is fouling, and someone has to "
        "wade into the cold channel with a rake before the pressure drops far enough to wake "
        "the supervisor. On the night the storm hit, the gallery went quiet and the crew "
        "stood listening until the backup generator caught and the pumps came back one after "
        "another.",
        100,
    ),
    "long": (
        "When the survey team reached the valley, the map they carried showed a lake "
        "where they found a dry basin of cracked clay, and the guide explained that the "
        "river had changed course three summers earlier after a landslide blocked the "
        "narrow pass above the village. Nobody had told the ministry, because the "
        "ministry had not asked, and the irrigation permits issued that spring were "
        "written for water that no longer existed. The team spent two days measuring "
        "the old shoreline, photographing the stumps of trees that had stood underwater "
        "for forty years, and interviewing farmers who remembered the lake as children. "
        "On the third morning a truck arrived with a portable pump and four engineers "
        "who had been sent to drain a marsh that was no longer there, and the guide "
        "laughed until he had to sit down on a rock and wipe his eyes twice. By evening "
        "the two groups had combined their notes, drawn a new channel on the map, and "
        "agreed to send a single honest report instead of the four contradictory ones "
        "their offices expected, which the team leader later called the only real "
        "achievement of the entire expedition, and the new channel still carries water "
        "today.",
        200,
    ),
}

SPEAKER_TEXT = (
    "<|speaker:0|> You said the shipment left on Tuesday morning. "
    "<|speaker:1|> It did, but the port held it for inspection. "
    "<|speaker:0|> Then why is it still standing in the yard? "
    "<|speaker:1|> Because that is exactly where the inspectors put it."
)
SPEAKER_CONTROL = SPEAKER_TEXT.replace("<|speaker:1|>", "<|speaker:0|>")


def connect(base, timeout):
    u = urlparse(base)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def post_whole(base, text, timeout=600, retries=20):
    """POST one whole WAV reply; TTFB = first body byte, total = last body byte."""
    body = json.dumps({
        "model": "fishaudio/drama-3",
        "input": text,
        "response_format": "wav",
    }).encode()
    attempt = 0
    while True:
        conn = connect(base, timeout)
        try:
            start = time.perf_counter()
            conn.putrequest("POST", "/v1/audio/speech", skip_accept_encoding=True)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            resp = conn.getresponse()
            head = {k.lower(): v for k, v in resp.getheaders()}
            first = resp.read(1)
            ttfb = time.perf_counter() - start
            rest = resp.read()
            total = time.perf_counter() - start
            data = first + rest
            if resp.status != 200:
                raise RuntimeError("HTTP %s: %s" % (resp.status, data[:300]))
            return {
                "status": resp.status,
                "ttfb_s": round(ttfb, 4),
                "wall_s": round(total, 4),
                "bytes": len(data),
                "audio_s": round(audio_seconds(len(data), WAV_HEADER), 3),
                "finish_reason": head.get("x-finish-reason"),
                "cfg_gamma": head.get("x-cfg-gamma"),
                "md5": hashlib.md5(data).hexdigest(),
                "wav": data,
            }
        except (ConnectionError, OSError) as exc:
            attempt += 1
            if attempt > retries:
                raise
            print("# retry after %r" % (exc,), file=sys.stderr)
            time.sleep(0.5)
        finally:
            conn.close()


def post_stream(base, text, timeout=600, out_path=None):
    """POST stream=true + response_format=pcm; time every block that arrives."""
    body = json.dumps({
        "model": "fishaudio/drama-3",
        "input": text,
        "response_format": "pcm",
        "stream": True,
    }).encode()
    conn = connect(base, timeout)
    try:
        start = time.perf_counter()
        conn.putrequest("POST", "/v1/audio/speech", skip_accept_encoding=True)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        heads = {k.lower(): v for k, v in resp.getheaders()}
        chunks, nbytes = [], 0
        sink = open(out_path, "wb") if out_path else None
        try:
            while True:
                block = resp.read(8192)
                if not block:
                    break
                now = time.perf_counter()
                nbytes += len(block)
                chunks.append((round(now - start, 4), len(block)))
                if sink:
                    sink.write(block)
        finally:
            if sink:
                sink.close()
        total = time.perf_counter() - start
        if resp.status != 200:
            raise RuntimeError("HTTP %s: stream failed" % resp.status)
        first_at, first_len = chunks[0] if chunks else (None, 0)
        return {
            "status": resp.status,
            "content_type": heads.get("content-type"),
            "transfer_encoding": heads.get("transfer-encoding"),
            "first_chunk_s": first_at,
            "first_chunk_bytes": first_len,
            "blocks": len(chunks),
            "bytes": nbytes,
            "audio_s": round(audio_seconds(nbytes), 3),
            "wall_s": round(total, 4),
            "gaps_s": [round(b[0] - a[0], 4) for a, b in zip(chunks, chunks[1:])][:8],
            "md5": hashlib.md5(open(out_path, "rb").read()).hexdigest() if out_path else None,
        }
    finally:
        conn.close()


def summarize(rows, key):
    vals = sorted(r[key] for r in rows)
    return {
        "median": round(statistics.median(vals), 4),
        "min": round(vals[0], 4),
        "max": round(vals[-1], 4),
        "runs": vals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18793")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    outdir = args.out
    os.makedirs(os.path.join(outdir, "stream"), exist_ok=True)
    report = {"base": args.base, "runs": args.runs, "started_epoch": time.time(), "texts": {}, "whole": {}, "stream": {}}

    for name, (text, claimed) in TEXTS.items():
        words = len(text.split())
        report["texts"][name] = {"claimed_words": claimed, "words": words, "chars": len(text), "text": text}

    # warm the engine, then measure
    for i in range(args.warmup):
        r = post_whole(args.base, TEXTS["short"][0])
        print("warmup %d: %.3fs wall, %.2fs audio" % (i + 1, r["wall_s"], r["audio_s"]), file=sys.stderr)

    for name in TEXTS:
        rows = []
        for run in range(1, args.runs + 1):
            r = post_whole(args.base, TEXTS[name][0])
            wav = os.path.join(outdir, "take-%s-%d.wav" % (name, run))
            if run == 1:
                with open(wav, "wb") as fh:
                    fh.write(r.pop("wav"))
                r["wav"] = wav
            else:
                r.pop("wav")
                os.remove(wav) if os.path.exists(wav) else None
            r["rtf"] = round(r["audio_s"] / r["wall_s"], 3) if r["wall_s"] else None
            rows.append(r)
            print("%-6s run %d: ttfb %.3fs wall %.3fs audio %.2fs rtf %.2f"
                  % (name, run, r["ttfb_s"], r["wall_s"], r["audio_s"], r["rtf"]), file=sys.stderr)
        report["whole"][name] = {
            "runs": rows,
            "ttfb_s": summarize(rows, "ttfb_s"),
            "wall_s": summarize(rows, "wall_s"),
            "audio_s": summarize(rows, "audio_s"),
        }

    for name in ("short", "medium", "hundred", "long"):
        pcm = os.path.join(outdir, "stream", "stream-%s.pcm" % name)
        r = post_stream(args.base, TEXTS[name][0], out_path=pcm)
        r["rtf"] = round(r["audio_s"] / r["wall_s"], 3)
        report["stream"][name] = r
        print("stream %-6s: first chunk %.3fs, wall %.3fs, audio %.2fs, %d blocks, rtfc %.2f"
              % (name, r["first_chunk_s"], r["wall_s"], r["audio_s"], r["blocks"],
                 r["audio_s"] / r["first_chunk_s"]), file=sys.stderr)

    # the streamed PCM must be the whole WAV's payload, byte for byte
    for name in ("medium", "hundred", "long"):
        wav = open(os.path.join(outdir, "take-%s-1.wav" % name), "rb").read()[WAV_HEADER:]
        pcm = open(os.path.join(outdir, "stream", "stream-%s.pcm" % name), "rb").read()
        report["stream"][name]["matches_wav_payload"] = hashlib.md5(wav).hexdigest() == hashlib.md5(pcm).hexdigest()
        report["stream"][name]["pcm_md5"] = hashlib.md5(pcm).hexdigest()
        print("stream %-6s matches whole-WAV payload: %s" % (name, report["stream"][name]["matches_wav_payload"]), file=sys.stderr)

    # two-speaker tags vs the same line with every tag on speaker 0
    tagged = post_whole(args.base, SPEAKER_TEXT)
    control = post_whole(args.base, SPEAKER_CONTROL)
    for tag, r in (("tagged", tagged), ("control", control)):
        with open(os.path.join(outdir, "take-speakers-%s.wav" % tag), "wb") as fh:
            fh.write(r.pop("wav"))
    report["speakers"] = {
        "tagged": {"wall_s": tagged["wall_s"], "audio_s": tagged["audio_s"], "md5": tagged["md5"], "finish_reason": tagged["finish_reason"]},
        "control_all_speaker_0": {"wall_s": control["wall_s"], "audio_s": control["audio_s"], "md5": control["md5"], "finish_reason": control["finish_reason"]},
        "identical_audio": tagged["md5"] == control["md5"],
    }
    # a malformed tag must be refused by the parser, not quietly spoken
    bad = json.dumps({"model": "fishaudio/drama-3", "input": "<|speaker:99|> out of range", "response_format": "wav"}).encode()
    conn = connect(args.base, 60)
    try:
        conn.request("POST", "/v1/audio/speech", body=bad, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        report["speakers"]["out_of_range_tag"] = {
            "status": resp.status,
            "body": resp.read()[:300].decode("utf-8", "replace"),
        }
    finally:
        conn.close()
    print("speakers: tagged %.2fs vs control %.2fs, identical=%s, out-of-range tag -> %s"
          % (tagged["wall_s"], control["wall_s"], report["speakers"]["identical_audio"],
             report["speakers"]["out_of_range_tag"]["status"]), file=sys.stderr)

    report["finished_epoch"] = time.time()
    with open(os.path.join(outdir, "bench.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["whole"], indent=2))


if __name__ == "__main__":
    main()
