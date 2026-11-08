# Installing Drama 3 (TensorFold native NVFP4) on a fresh Omarchy box

This is the build that was measured on `ripper` — see [`BENCH-5090.md`](BENCH-5090.md) for the
numbers. Everything below was executed on that machine on **2026-11-08**; the versions and
paths are the ones in use, not a guess. End state: one RTX 5090 serving
`POST /v1/audio/speech` with model id `fishaudio/drama-3` on `127.0.0.1:18793`,
**20.5 GiB VRAM**, ~5.8× real time, ready in ~10 s after start.

Target: Omarchy (Arch) on a single **RTX 5090** (sm_120), CUDA 13, user systemd. The host
also has an RTX 3090 installed; it is excluded with `CUDA_VISIBLE_DEVICES=0` (see gotcha 11).

## 1. Packages (exact pacman names)

Versions as installed on `ripper`:

```sh
sudo pacman -S --needed \
  cuda                    # 13.4.3-1  -> /opt/cuda/bin/nvcc (V13.4.92); NOT on PATH by default
  zig                     # 0.17.0-1  -> /usr/bin/zig; must match the engine's .zig-version exactly
  python                  # 3.14.8-1  -> /usr/bin/python3, for recipe/pilot.py and the harnesses
  python-huggingface-hub  # 1:2.2.0-1 -> /usr/bin/hf, downloads the weights
  git                     # 2.56.0-1  -> clones the TensorFold engine at its pinned commit
  util-linux              # flock, fuser (used by scripts/drama3-service.sh)
  curl                    # 8.23.0-1  -> /health checks and the repro commands
  ffmpeg                  # 2:9.0.2-2 -> ffprobe/ffmpeg, only to inspect and resize the WAVs
```

NVIDIA userspace is already on an Omarchy desktop; the versions used here were
`nvidia-open-dkms 615.78.08-1`, `nvidia-utils 615.78.08-1`, `lib32-nvidia-utils 615.78.08-1`.
`nvidia-smi` must report the 5090 as compute capability sm_120. Kernel: `7.2.15-1-omarchy`.

Sizes and headroom, from the launcher's guard table:

| Quantity | Value |
|---|---|
| NVFP4 weights + codec on disk | 20.76 GiB (22,294,649,076 B, 23 files, 21 GB `du`) |
| Engine checkout + build output | 379 MB (`build/tensorfold-nvfp4`) |
| Free RAM required before load (`MemAvailable`) | 88 GiB |
| cgroup cap on the server unit (`MemoryMax`), swap off | 54 GiB |
| Host headroom floor / abort threshold | 30 GiB / 34 GiB |
| Measured host cost after load | 20,490 MiB VRAM, 676 MiB RSS |

## 2. Get the recipe and build the engine

```sh
mkdir -p /home/jun/drama3 && cd /home/jun/drama3
git clone https://github.com/plotarmordev/drama-tensorfold.git recipe-src
cd recipe-src
git checkout f3b1270            # the checkout these numbers came from

# nvcc is in /opt/cuda/bin and is NOT on PATH; pass it explicitly.
ZIG=/usr/bin/zig NVCC=/opt/cuda/bin/nvcc scripts/build-engine.sh --weights nvfp4
```

What the script does (no manual steps): checks `engine/nvfp4/UPSTREAM`
(`plotarmordev/TensorFold` @ `b1f4b9b4`), verifies `engine/nvfp4/patches/SHA256SUMS`, clones
the engine into `build/tensorfold-nvfp4`, `git am`s the four patches (0001 protected speech
chunks, 0002 speech sampling controls, 0003 dialogue speakers, 0004 redraw salt), refuses a
Zig version that is not the pinned `0.17.0`, then runs
`zig build native -Dnvcc=/opt/cuda/bin/nvcc --summary all`.

Observed output tail:

```
Build Summary: 49/49 steps succeeded
native success
+- install tensorfold-native success
   +- compile exe tensorfold-native safe native success 2m MaxRSS:1G
engine (nvfp4 weights): b1f4b9b4830c9bd43bf7c432cffe7dea21313537 + 4 patches = d6b805a233c7e21357c396449432c49214505ff0
64fa0dc12272ece450bba566a37c67599e0ce0479a3dd23d203ba29e69a885db  zig-out/native/bin/tensorfold-native
```

The build compiles ~40 CUDA fatbins with `-gencode` for sm_86 / sm_120 / sm_121; both
successful builds on this box are in `/home/jun/drama3/logs/build-engine.log` and
`build-sm120.log` (the second run is fully cached: `options cached`, `(reused)`). Ignore the
two nvcc warnings about unused `SMEM_MAX` / `RW` constants — they are in the pinned upstream.

## 3. Weights

Source: **`plotarmordev/drama-3-NVFP4-Pruned`** on Hugging Face, revision
**`494db08481db952f834b2241bce63fb3dfa72554`**, 23 files, 22,294,649,076 bytes
(20.76 GiB on disk; 19.05 GiB of model tensors in VRAM). License: Fish Audio Community
License 1.0 — read `drama-3-NVFP4-Pruned/LICENSE` before serving anyone but yourself
(serving third parties can be a "Hosted Service" under Section III(c)).

`recipe/pilot.py` downloads it on first start:

```sh
cd /home/jun/drama3/recipe-src
python3 recipe/pilot.py --weights nvfp4 \
  --runtime-dir "$PWD/runtime" --gpu-lock /tmp/gpu.lock \
  --evidence-dir runs/pilot-02 --reserve-gib 8
```

Observed on this box: `Fetching 23 files … 3:00` (~3 min), then
`Checking drama-3-NVFP4-Pruned against recipe/checksums/nvfp4.sha256 … matches (13 s)`,
then `DRAMA_PILOT_READY`. The download prints `Warning: You are sending unauthenticated
requests to the HF Hub` — harmless; set `HF_TOKEN` if you hit rate limits.

Manual equivalent (if you do not want the launcher):

```sh
cd /home/jun/drama3/recipe-src
hf download plotarmordev/drama-3-NVFP4-Pruned \
  --revision 494db08481db952f834b2241bce63fb3dfa72554 \
  --local-dir drama-3-NVFP4-Pruned
```

Every launcher start re-hashes the weights **before** taking the GPU lock and refuses missing
or changed files. A direct `serve` (what `scripts/drama3-service.sh` does) does **not** hash;
run `sha256sum -c recipe/checksums/nvfp4.sha256` yourself if you need that guarantee after a
disk event.

## 4. Serve

The flags that were measured (NVFP4 guidance γ 1.5 + batched guidance is the setting the quant
was validated with; `--no-drafts`, no prompt cache and no keep-warm for a single-synthesis box):

```sh
bin=build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native
ckpt=drama-3-NVFP4-Pruned
$bin serve $ckpt \
  --host 127.0.0.1 --port 18793 --name fishaudio/drama-3 \
  --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0 \
  --cfg-gamma 1.5 --cfg-batched
```

Run it under the same environment the launcher used (this is what the control script sets):

```
CUDA_VISIBLE_DEVICES=0          # 5090 is card 0; the 3090 must stay out
HF_HOME=<runtime>/hf-cache
HF_HUB_OFFLINE=1                # only after the weights are downloaded
XDG_CACHE_HOME=<runtime>/runtime-cache
TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / CUDA_CACHE_PATH / FLASHINFER_WORKSPACE_BASE
    = <runtime>/runtime-cache/*
TENSORFOLD_MEMORY_RESERVE_GIB=8
TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54
```

Measured startup, from the server's own log:

```
[tensorfold] serving fishaudio/drama-3 at http://127.0.0.1:18793/v1
             (sampling: greedy; drafts: off; context: 8192; loaded in 9.4s)
```

## 5. Control it with `scripts/drama3-service.sh`

Copy it to the box and use it as the only start/stop path (it is idempotent, so a UI toggle
can call it blindly):

```sh
install -m 755 scripts/drama3-service.sh /home/jun/drama3/scripts/drama3-service.sh
/home/jun/drama3/scripts/drama3-service.sh start     # exit 0; refuses (2) if :18793 already answers
/home/jun/drama3/scripts/drama3-service.sh status    # exit 0 answering, 1 not, 2 foreign server
/home/jun/drama3/scripts/drama3-service.sh restart
/home/jun/drama3/scripts/drama3-service.sh stop      # exit 0; refuses (2) if the server is not ours
```

It starts a transient user unit `drama3-service.service` whose PID holds
`flock` on **both** `/tmp/drama3-gpu.lock` (created by the script) and the host's shared
`/tmp/gpu.lock` for the server's whole lifetime, so no other GPU job can collide with it.
`stop` only stops that unit, and only after verifying its `MainPID` is this binary on this
port. Exit codes: 0 ok, 1 failure, 2 someone else owns the port, 3 GPU lock held elsewhere,
4 state mismatch. Verbatim test runs are in
[`BENCH-5090.md` §7](BENCH-5090.md#7-control-script--all-four-verbs-tested-on-this-box).

## 6. Verify

```sh
curl -sS http://127.0.0.1:18793/health -o /dev/null -w '%{http_code}\n'     # 200
curl -sS http://127.0.0.1:18793/v1/models | jq -r '.data[0].id'             # fishaudio/drama-3
curl -sS http://127.0.0.1:18793/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"The train leaves at seven fifteen.","response_format":"wav"}' \
  -o hello.wav -w 'ttfb=%{time_starttransfer}s total=%{time_total}s\n'
ffprobe -v error -show_entries format=duration:stream=sample_rate,channels -of default=nw=1 hello.wav
# duration≈3.5  sample_rate=44100  channels=1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # tensorfold-native, 20490 MiB
```

Full benchmark harness (stdlib only, starts nothing): `scripts/bench-drama3.py`;
speaker-tag proof: `scripts/check-speaker-tags.py`.

## 7. Gotchas — each one hit on this box

1. **`nvcc` is not on PATH.** `which nvcc` fails on Omarchy; the Arch `cuda` package installs
   to `/opt/cuda/bin`. Pass `NVCC=/opt/cuda/bin/nvcc` to `build-engine.sh` (or prepend
   `/opt/cuda/bin` to PATH). The launcher's `PATH=/usr/local/cuda/bin:...` is a leftover and
   matches nothing.

2. **Zig must be exactly 0.17.0.** The engine's `.zig-version` pins it and the build script
   refuses anything else (`Zig <v> found; the engine pins 0.17.0`). Arch's `zig` was 0.17.0-1.

3. **The launcher's default memory reserve fails on this host.** `pilot.py` defaults to
   `--reserve-gib 34`, and the engine rejects it:

   ```
   error: Drama memory admission: TENSORFOLD_MEMORY_RESERVE_GIB=34: a number of GiB from 0.5 to the memory's size
   tensorfold: the native Drama 3 CUDA engine could not load the checkpoint (DramaMemoryAdmission)
   Service runtime: 494ms ... status=1/FAILURE
   ```

   That is `runs/pilot-01`. **Use `--reserve-gib 8`** (pilot-02 and everything measured here).

4. **Every `pilot.py` start pays a full re-hash of 20.76 GiB (~13 s)** before the GPU lock is
   taken. A direct `serve` skips it. Budget ~20 s from `pilot.py` invocation to `DRAMA_PILOT_READY`;
   `scripts/drama3-service.sh` starts the engine in **10 s** (no hashing).

5. **The reply ceiling is 1024 frames = 47.554 s.** A ~200-word text is clipped and returns
   `X-Finish-Reason: length` (see `BENCH-5090.md` §1). Split long text client-side or raise
   `--max-frames` (not used for the published numbers).

6. **`voice` must be `default`** (or a catalog loaded with `--voices`). OpenAI SDKs send
   `alloy` by default and get:

   ```
   refused ...: Drama 3 request: unknown voice 'alloy'; only the default voice is supported
   ```

7. **Reference audio is not accepted inline.** `ref_audio` → `ref_audio is not supported by
   this model`; `reference_audio` → `unknown speech request field: reference_audio`; a
   `references.0` object must carry codec codes:
   `reference audio is not encoded by this server; send its vq_codes (tools/zig/drama_voice.py)`.
   Encode voice clips with `tools/zig/drama_voice.py` and serve them with `--voices`.

8. **Streaming is PCM only, over HTTP/1.1.** `stream: true` with `response_format: "wav"` is
   refused (`stream=true requires response_format='pcm'`), `stream_format: "sse"` is refused,
   and `stream=true` is rejected on HTTP/1.0. The client also has to keep reading: after the
   terminating chunk the connection stays open (keep-alive), so a naive `recv()` loop must
   stop on the chunked terminator or set a timeout.

9. **Speaker tags are parsed, and a bad one is a 400, not audio.** `<|speaker:0|>` …
   `<|speaker:15|>` work (alternating voices, see `BENCH-5090.md` §3). `<|speaker:99|>`,
   `<|speaker:01|>`, `<|speaker:|>`, `<|speaker0|>` are refused with
   `speaker tags must be written <|speaker:N|> with N in 0..15 and no leading zero`.
   A tag only counts when a tag is what leads the text; otherwise the builder prefixes
   `<|speaker:0|>`.

10. **A lone `<|speaker:1|>` take is very quiet.** Same sentence as `speaker:0`: peak 0.044 vs
    0.94 (−27 dBFS). Inside a multi-turn dialogue the same voice is the loudest one. Do not
    use a speaker-1-only request to judge levels; normalise in the player or supply a reference
    voice.

11. **Two GPUs: pin the 5090.** `ripper` also has a 3090, which is *not* used
    (`CUDA_VISIBLE_DEVICES=0`, engine reports `CUDA sm_120 device 0 ... RTX 5090`). The binary
    carries sm_86/sm_120/sm_121 fatbins, so without the pin it would start on the 3090 too —
    slower, with a third of the VRAM, and the timings in `BENCH-5090.md` would not apply.

12. **One synthesis at a time.** `--parallel 1`; the HTTP contract returns `503` while the
    engine is busy (documented in the recipe README; the harnesses retry on 503). Serialise
    callers or queue upstream.

13. **`serve --help` looks like an LLM server, and the log says "thinking on".** The native
    binary is shared with TensorFold's text server: it advertises `--thinking`,
    `--chat-template`, `--drafter` and prints
    `thinking on (the chat template's default) ...`. None of that affects
    `/v1/audio/speech`. Ignore it.

14. **cuBLASLt note on sm_120** — the engine logs this at startup (informational, but it is a
    real fallback on this card):

    ```
    info: drama tower: one-row projections on cuBLASLt: drama_tower_gemv is qualified on
    sm 121 with cuBLASLt 130101, not sm 120 with 130800
    ```

    One-row projections therefore take the generic cuBLASLt path on this card. Nothing to fix
    from this side; it is recorded so nobody chases it as a misconfiguration.

15. **Build reproducibility details.** `build-engine.sh` refuses a build directory with local
    changes and re-applies the whole patch series each run (`git am` with fixed committer data),
    so a rebuild produces the same patched HEAD `d6b805a2`. A rebuild with unchanged sources is
    cached and takes seconds. Clone time + build ≈ 5 min; peak RSS ~1 GiB during the nvcc steps.

16. **The GPU lock is a host convention, not a driver feature.** The pilot used the host's
    `/tmp/gpu.lock` (held by the launcher process, pid 14220, for the server's lifetime).
    `scripts/drama3-service.sh` takes both `/tmp/drama3-gpu.lock` and `/tmp/gpu.lock` so it
    inter-operates with every other job on the box; if your host uses a different shared lock,
    set `DRAMA3_HOST_GPU_LOCK` to it.

17. **Loopback only.** Keep `--host 127.0.0.1`. Exposing the port makes the model a Hosted
    Service under the Fish Audio Community License; that needs a separate licence from Fish Audio.

## 8. Teardown

```sh
/home/jun/drama3/scripts/drama3-service.sh stop
# remove the engine + weights (21 GB)
rm -rf /home/jun/drama3/recipe-src/drama-3-NVFP4-Pruned /home/jun/drama3/recipe-src/build/tensorfold-nvfp4
```
