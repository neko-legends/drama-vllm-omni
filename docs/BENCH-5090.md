# Drama 3 on one RTX 5090 — measured benchmark

All numbers below were produced on **2026-11-08** (Sun, PST / UTC−08:00) on `ripper`,
Jun's RTX 5090 box, against a **warm** server that had already answered requests.
Every row was run **3 times**; median and min are reported. Nothing here is copied from
another card's results.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB (GB202, `10de:2b85`), compute capability sm_120 |
| Driver | 615.78.08 (nvidia-open-dkms), VBIOS 98.02.2E.00.AF |
| Second GPU | RTX 3090 (idle, not used: 109 MiB) |
| Host | Omarchy (Arch), kernel 7.2.15-1-omarchy, 503 GiB RAM, user systemd |
| Engine | TensorFold native Zig/CUDA NVFP4, `plotarmordev/TensorFold` `b1f4b9b4` + 4 patches = `d6b805a2` |
| Weights | `plotarmordev/drama-3-NVFP4-Pruned` @ `494db084`, 22,294,649,076 B (20.76 GiB) |
| Recipe checkout | `plotarmordev/drama-tensorfold` @ `f3b1270` in `/home/jun/drama3/recipe-src` |
| Audio | WAV 44.1 kHz, 16-bit, mono (`pcm_s16le`) |
| Sampling | greedy — the same input returned byte-identical WAVs on all 3 runs |
| Server warm-up | 2 discarded requests before each table |

## Serve command line (as measured)

```
/home/jun/drama3/recipe-src/build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native \
  serve /home/jun/drama3/recipe-src/drama-3-NVFP4-Pruned \
  --host 127.0.0.1 --port 18793 --name fishaudio/drama-3 \
  --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0 \
  --cfg-gamma 1.5 --cfg-batched
```

Environment the server runs with (the recipe's guard, unchanged):

```
CUDA_VISIBLE_DEVICES=0  HF_HOME=<runtime>/hf-cache  HF_HUB_OFFLINE=1
TENSORFOLD_MEMORY_RESERVE_GIB=8  TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54
XDG_CACHE_HOME / TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / CUDA_CACHE_PATH /
FLASHINFER_WORKSPACE_BASE = <runtime>/runtime-cache/...
```

The engine's own startup line (from the server log) states the operating point:

```
Drama 3 on CUDA sm_120 device 0 (NVIDIA GeForce RTX 5090): models 19.05 GiB; context 8192 tokens,
prompts up to 4096 tokens in passes of 128 rows, up to 1024 frames of 2048 samples at 44100 Hz;
routed experts held 7369 of 10240; one synthesis at a time; routed experts NVFP4 in 40 of 40 layers;
guidance on: a second backbone sequence of 0.304 GiB, run every 1 frames in one batch with the first
```

Unit start → `GET /health` 200 measured **10 s** (three separate starts via
`scripts/drama3-service.sh start`, which the benchmark used for its last measurements).
`recipe/pilot.py`'s own evidence file reports `cold_ready_s: 9.42`.

## 1. Whole reply (`response_format: wav`)

A whole WAV is only written after the take is complete, so time-to-first-byte and total
wall time are the same number here (they differ by < 3 ms). TTFB = first response body byte
as seen by the client; wall = request write → last body byte. RTF = generated audio seconds /
wall seconds.

| Input | Words | Chars | TTFB median | TTFB min | Wall median | Wall min | Audio s | RTF median | RTF min | `X-Finish-Reason` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short | 10 | 54 | 0.5951 s | 0.5946 s | 0.5952 s | 0.5949 s | 3.483 | **5.85×** | 5.845× | `stop` |
| medium | 50 | 291 | 3.818 s | 3.8176 s | 3.820 s | 3.8196 s | 22.848 | **5.98×** | 5.981× | `stop` |
| 93-word (largest reply that finished) | 93 | 501 | 4.814 s | 4.8135 s | 4.816 s | 4.8155 s | 28.607 | **5.94×** | 5.940× | `stop` |
| 202-word | 202 | 1156 | 8.165 s | 8.163 s | 8.168 s | 8.1667 s | 47.554 | **5.82×** | 5.821× | `length` (clipped) |

Raw runs, in order:

```
short   TTFB 0.5951 / 0.5957 / 0.5946    wall 0.5952 / 0.5959 / 0.5949   audio 3.483   RTF 5.852 / 5.845 / 5.855
medium  TTFB 3.8176 / 3.8176 / 3.8182    wall 3.8199 / 3.8196 / 3.8202   audio 22.848  RTF 5.981 / 5.982 / 5.981
93-word TTFB 4.8135 / 4.8136 / 4.8139    wall 4.8160 / 4.8155 / 4.8162   audio 28.607  RTF 5.940 / 5.941 / 5.940
202-word TTFB 8.1630 / 8.1670 / 8.1646   wall 8.1667 / 8.1692 / 8.1682   audio 47.554  RTF 5.823 / 5.821 / 5.822
```

Determinism: all three takes in every row are byte-identical, so the spread above is
scheduling noise, not sampling variance. WAV md5 per row:
`37f7666409da88705e2a67358652cc85` (10 w), `7e70939d98c1717c77c328af265aa27c` (50 w),
`9ad22eb7f87831155ee2645aefd01a59` (93 w), `ac02351e7617b3ab3032052091509dd4` (202 w).

**The 202-word row is capped, not complete.** The engine answers at most 1024 frames of
2048 samples = **47.554 s** per request, so a 202-word text comes back clipped with
`X-Finish-Reason: length` (the log line says `finish=length`). Its 5.82× is sustained
generation throughput; the full text was never spoken. At this voice's pace (0.31–0.46 s of
audio per word in these samples) the cap lands at roughly 100–150 words, so the 93-word row is the largest
complete reply measured end to end. Longer texts must be split client-side (or the cap raised
with `--max-frames`, which was not used for these numbers).

## 2. Streamed reply (`response_format: pcm`, `stream: true`, HTTP/1.1 chunked)

`stream=true` is refused unless `response_format` is `pcm`; the body is raw `pcm_s16le`
44.1 kHz mono in HTTP/1.1 chunks. "First chunk" is the first **body** block my client read;
`first_audio_ready` is the engine's own log field (frame 0 ready inside the server), which is
the floor the network and client cannot beat. The ~30 ms between them is client read
granularity on this box, not an extra GPU cost.

| Input | First chunk (client) | `first_audio_ready` (server log) | Wall s | Audio s | Chunks | RTF |
|---|---:|---:|---:|---:|---:|---:|
| short (10 w) | **0.064 s** | 0.030 s | 0.657 | 3.483 | 38 | 5.30× |
| medium (50 w) | **0.072 s** | 0.038 s | 4.225 | 22.848 | 246 | 5.41× |
| 93-word | **0.080 s** | 0.045 s | 5.326 | 28.607 | 308 | 5.37× |
| 202-word | **0.113 s** | 0.079 s | 9.014 | 47.554 | 512 | 5.28× |

The streamed body is byte-identical to the whole WAV's payload (after its 44-byte header) —
verified for the medium, 93-word and 202-word inputs (`matches_wav_payload: true` in
`bench/bench.json`). Streaming therefore costs nothing in fidelity, and the first audio reaches
a player ~0.07 s after the request while the rest is generated ~5.3× faster than it plays.

Delivery cadence, measured on a 93-word stream with a raw socket (155 chunks,
2,523,136 B = 28.607 s of audio): one chunk of ~16.3 kB (~185 ms of audio) every
**34.3 ms** (median; max gap 35.1 ms, p90 34.8 ms) starting at **0.0459 s**. Playout keeps
a lead of many seconds after the first chunk.

## 3. Two-speaker tags — they work

`<|speaker:N|>` with N in 0..15 is parsed, validated and applied. Evidence, all measured:

Same sentence spoken under each tag (the engine's prompt builder leaves a leading tag alone):

| Take | Audio s | Median f0 | Peak | WAV md5 |
|---|---:|---:|---:|---|
| `<\|speaker:0\|> Good evening, the harbour lights are coming on.` | 2.368 | 162.7 Hz | 0.94 | `58eec879ee3b9f67a5b9f63924ecd9f3` |
| `<\|speaker:1\|> Good evening, the harbour lights are coming on.` | 1.997 | ~101 Hz (low confidence: −27 dBFS take) | 0.044 | `d7cf557dc266e51d6070045c953e0e90` |
| the `speaker:0` take again | 2.368 | 162.7 Hz | 0.94 | `58eec879ee3b9f67a5b9f63924ecd9f3` (identical) |

So the tag changes the voice (different bytes, different register) and the same tag is
reproducible. A lone take whose first tag is `speaker:1` came back very quiet (−27 dBFS,
peak 0.044 vs 0.94); that is a property of this built-in voice with no preceding speaker-0
context, not a tag failure — inside a dialogue the same voice is loud (below).

Four-turn dialogue (`<|speaker:0|>` / `<|speaker:1|>` alternating), split at its silences and
pitched per turn:

| Turn | Span | Median f0 | RMS |
|---|---|---:|---:|
| 1 — speaker 0 | 0.07–2.36 s | 110.4 Hz | 0.146 |
| 2 — speaker 1 | 3.62–6.77 s | 253.4 Hz | 0.205 |
| 3 — speaker 0 | 8.02–10.14 s | 119.0 Hz | 0.142 |
| 4 — speaker 1 | 11.47–14.95 s | 247.8 Hz | 0.216 |

Alternating ~110 Hz / ~250 Hz across the turns is the two voices actually swapping, i.e. the
tags are honoured, not stripped. The same text with every tag rewritten to `speaker:0`
produced different audio and a different result (`X-Finish-Reason: silence`, 12.725 s vs
15.325 s) — a like-for-like control is not possible because the one-voice version trips the
silence guard; the per-turn pitch table above is the reliable evidence.

Malformed tags are refused, not spoken:

```
$ curl -sS http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"model":"fishaudio/drama-3","input":"<|speaker:99|> out of range","response_format":"wav"}'
{"error": {"message": "Drama 3 request: input: speaker tags must be written <|speaker:N|> with N in 0..15
 and no leading zero; got '<|speaker:99|> out of range'", "type": "invalid_request_error"}}
# HTTP 400
```

## 4. VRAM and host cost

```
$ nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
index, name, driver_version, memory.used [MiB], memory.total [MiB]
0, NVIDIA GeForce RTX 5090, 615.78.08, 22339 MiB, 32607 MiB
1, NVIDIA GeForce RTX 3090, 615.78.08, 109 MiB, 24576 MiB

$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
...
14301, .../tensorfold-native, 20490 MiB

$ systemctl --user show drama3-pilot-...service -p MemoryCurrent
MemoryCurrent=708919296        # 676 MiB RSS
```

- **Engine: 20,490 MiB VRAM** (the other ~1.8 GiB on card 0 is the desktop).
- VRAM did not move during synthesis: sampling every 0.2 s across a 47.55 s/8.17 s
  request gave `peak_during_request = 20,490 MiB`, same as idle.
- Host RSS 676 MiB; cgroup cap `MemoryMax=54 GiB`, swap off, weights+buffers capped at 54 GiB.

## 5. Reproduce — exact commands

Everything ran on `ripper`, against `127.0.0.1:18793`. Whole reply, with the timings the
table above reports (`curl`'s `time_starttransfer` is the first response byte):

```sh
S='The train leaves at seven fifteen and we should hurry.'

curl -sS -D headers.txt -o take.wav \
  -w 'http=%{http_code} ttfb=%{time_starttransfer}s total=%{time_total}s bytes=%{size_download}\n' \
  http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"fishaudio/drama-3\",\"input\":\"$S\",\"response_format\":\"wav\"}"
# http=200 ttfb=0.616325s total=0.616770s bytes=307244  (a run after the restart in §7;
#                                                     the warm medians are the table in §1)
grep -iE 'x-finish-reason|x-cfg-gamma|content-type' headers.txt
# Content-Type: audio/wav
# X-Finish-Reason: stop
# X-Cfg-Gamma: 1.5
ffprobe -v error -show_entries format=duration:stream=sample_rate,channels,bits_per_sample \
  -of default=nw=1 take.wav
```

Streamed reply (first byte here is the response line; the first **audio** block follows
~15–35 ms later, see the raw-socket number in §2):

```sh
curl -sSN -D headers-stream.txt -o stream.pcm \
  -w 'http=%{http_code} first_byte=%{time_starttransfer}s total=%{time_total}s bytes=%{size_download}\n' \
  http://127.0.0.1:18793/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"fishaudio/drama-3\",\"input\":\"$S\",\"response_format\":\"pcm\",\"stream\":true}"
# http=200 first_byte=0.030186s total=0.653915s bytes=307200
# Content-Type: audio/pcm ; Transfer-Encoding: chunked
python3 -c 'import os; print("audio_s", os.path.getsize("stream.pcm")/2/44100)'
# audio_s 3.483
```

Repeatable harnesses (stdlib only, they start and stop nothing):

```sh
scp scripts/bench-drama3.py scripts/check-speaker-tags.py ripper:/tmp/
ssh ripper 'python3 /tmp/bench-drama3.py --out /tmp/drama3-bench --runs 3'
ssh ripper 'python3 /tmp/check-speaker-tags.py --out /tmp/drama3-speakers'
```

`bench-drama3.py` writes `bench.json` (every run, every md5) plus the takes;
`check-speaker-tags.py` writes `speakers.json` and the four speaker takes. The copies
committed here are `bench/bench.json` and `bench/speakers.json`, taken from this run.

## 6. Auditable audio in this repo

Three takes, kept under 1 MB each, referenced by the tables:

| File | What it is | Duration / rate | md5 |
|---|---|---|---|
| `bench/audio/take-10w.wav` | the 10-word row, untouched 44.1 kHz take | 3.483 s / 44.1 kHz | `37f7666409da88705e2a67358652cc85` |
| `bench/audio/take-50w-16k.wav` | the 50-word row, decimated from the 44.1 kHz original for size | 22.848 s / 16 kHz | `f46ca502c183c655cc0ccb6535f16da0` |
| `bench/audio/take-2speaker-16k.wav` | the four-turn two-speaker take, decimated for size | 15.325 s / 16 kHz | `c20f2cbadbe8d48cd98948e13193840d` |

The two 16 kHz files were produced from the originals with:

```sh
ffmpeg -i take-medium-1.wav -ar 16000 -ac 1 -c:a pcm_s16le take-50w-16k.wav
ffmpeg -i take-speakers-tagged.wav -ar 16000 -ac 1 -c:a pcm_s16le take-2speaker-16k.wav
```

Audio duration, RTF and every timing above come from the untouched 44.1 kHz originals; the
downsample only changes what is stored here. The original md5s are in `bench/bench.json`
(`whole.<row>.runs[0].md5`, `speakers.tagged.md5`).

## 7. Control script — all four verbs tested on this box

`scripts/drama3-service.sh` (also installed at `/home/jun/drama3/scripts/drama3-service.sh`)
was tested against the running pilot first, then took over the port. Verbatim output:

```
### ./drama3-service.sh status          # the pre-existing pilot, not started by this script
state:          foreign-server
unit:           drama3-service.service (inactive)
port:           http://127.0.0.1:18793 answering
pid:            14301
vram:           20490 MiB
lock:           /tmp/gpu.lock held by pid 14220
cmdline:        .../tensorfold-native serve .../drama-3-NVFP4-Pruned --host 127.0.0.1 --port 18793 ...
drama3-service: note: a server this script did not start owns the port; start/stop will refuse
exit=0

### ./drama3-service.sh start
drama3-service: refusing to start: http://127.0.0.1:18793 already answers (pid 14301: ...tensorfold-native serve ...)
exit=2

### ./drama3-service.sh stop
drama3-service: refusing to stop: http://127.0.0.1:18793 is answered by pid 14301, which this script did not start
exit=2
```

The pilot was then stopped cleanly (`systemctl --user stop drama3-pilot-14220-…service`;
server and launcher gone, `/tmp/gpu.lock` free, port closed) and the script's own unit tested:

```
### ./drama3-service.sh status
state:          stopped
unit:           drama3-service.service (inactive)
port:           http://127.0.0.1:18793 closed
lock:           /tmp/gpu.lock free
exit=1

### ./drama3-service.sh start
Running as unit: drama3-service.service; invocation ID: daa57cce414d4d2dac54e9003850a2ed
drama3-service: started unit drama3-service.service; log /home/jun/drama3/logs/service.log
drama3-service: ready on http://127.0.0.1:18793 after 10s (unit drama3-service.service, pid 257808)
exit=0

### ./drama3-service.sh start            # idempotent
drama3-service: already running (unit drama3-service.service, pid 257808) — nothing to do
exit=0

### ./drama3-service.sh status
state:          running
unit:           drama3-service.service (active)
port:           http://127.0.0.1:18793 answering
pid:            257808
vram:           20490 MiB
lock:           /tmp/drama3-gpu.lock held by pid 257808
lock:           /tmp/gpu.lock held by pid 257808
cmdline:        .../tensorfold-native serve ... --port 18793 --name fishaudio/drama-3 --context 8192
                --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0 --cfg-gamma 1.5 --cfg-batched
exit=0
```

```
### ./drama3-service.sh restart
drama3-service: stopped unit drama3-service.service (pid 257808); http://127.0.0.1:18793 no longer answers
Running as unit: drama3-service.service; invocation ID: 2b0388cc202145f5967355512b9c5693
drama3-service: started unit drama3-service.service; log /home/jun/drama3/logs/service.log
drama3-service: ready on http://127.0.0.1:18793 after 10s (unit drama3-service.service, pid 258304)
exit=0

### ./drama3-service.sh stop
drama3-service: stopped unit drama3-service.service (pid 258304); http://127.0.0.1:18793 no longer answers
exit=0

### ./drama3-service.sh stop              # idempotent
drama3-service: not running — nothing to do
exit=0

### ./drama3-service.sh status
state:          stopped
unit:           drama3-service.service (inactive)
port:           http://127.0.0.1:18793 closed
lock:           /tmp/drama3-gpu.lock free
lock:           /tmp/gpu.lock free
exit=1

### ./drama3-service.sh start
drama3-service: ready on http://127.0.0.1:18793 after 10s (unit drama3-service.service, pid 258635)
exit=0
```

The GPU-lock guard, tested by holding `/tmp/drama3-gpu.lock` from another process:

```
### ./drama3-service.sh start
drama3-service: GPU lock /tmp/drama3-gpu.lock is held by pid 259128 — refusing to start
exit=3

### ./drama3-service.sh start            # after the holder exited
drama3-service: ready on http://127.0.0.1:18793 after 10s (unit drama3-service.service, pid 259340)
exit=0
```

The engine restarted by the script is the same engine: after `restart`, the 10-word request
returned md5 `37f7666409da88705e2a67358652cc85`, byte-identical to the pre-restart benchmark take.

## 8. What to take away

- One RTX 5090 speaks this model at **~5.8–6.0× real time**, warm, greedy, one take at a time.
  A 10-word line is **0.60 s** end to end; a 50-word paragraph is **3.8 s**.
- **Streaming is the interactive win**: first audio is ready in the engine at **30–45 ms**,
  reaches a client at ~65–110 ms, and then arrives ~5.3× faster than it plays, so playback
  never catches up with generation after the first second.
- **Two-speaker tags work** (0..15), with alternating ~110 Hz / ~250 Hz voices in the
  four-turn take; out-of-range tags are refused with HTTP 400.
- The 1024-frame (**47.55 s**) ceiling is the one hard limit to design around; a 202-word
  input is clipped, not truncated by the client.
- VRAM is flat at **20.5 GiB**, unaffected by request length.
