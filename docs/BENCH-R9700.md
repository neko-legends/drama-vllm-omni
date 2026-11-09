# Drama 3 on a 32 GB RDNA4 card (gfx1201) — working recipe and measurements

**Date:** 2026-11-08 · **Host:** the assistant host workstation · **GPU:** AMD Radeon AI PRO R9700, 32 GB, `gfx1201`, ROCm 7.2.5
**Runtime:** vLLM 0.33.0 (ROCm wheel, bundled torch 2.13.0+rocm10.0.0) + vLLM-Omni `0.1.dev1+g62876f40a`
**Weights:** `fishaudio/drama-3-int4` (24.0 GB on disk) — the FP8 checkpoint is 36.9 GiB of stage-0 weights and does **not** fit 32 GB.

Result: it **works**, at **1.3–1.9× faster than realtime** and **0.29 s to first audio** when streaming.

---

## Why this page exists

Every published Drama 3 path assumes either an H200/A100 (CUDA) or a GB10 Spark (TensorFold). Nothing documents
running it on a 32 GB RDNA4 Radeon, and the two things that block you are not obvious:

1. the INT4 checkpoint's zero-points are laid out in a way that trips a **stride assertion** in vLLM's RDNA hybrid
   W4A16 kernel, and
2. there are **two identical 32 GB cards** on this host, and the first one is already occupied by the assistant host's
   embedding/rerank/classifier services.

---

## The blocker, and the fix

Loading the INT4 checkpoint dies during stage-0 init with:

```
AssertionError: zp row stride 1536 must match scales 16
  at vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:388
```

That kernel (`RDNAHybridW4A16LinearKernel`) expects the zero-points packed as `[N//8, K//G]` with the same row stride as
the scales. The `drama-3-int4` checkpoint stores them per row, so the strides never match.

**Fix — disable that one kernel and let vLLM fall back to its other W4A16 kernels:**

```bash
export VLLM_DISABLED_KERNELS=RDNAHybridW4A16LinearKernel
```

With it disabled the loader picks the next kernel in the ROCm list and the model loads and serves. No patch to vLLM,
no fork. (`--linear-backend` can force the same choice; the env var is what we shipped.)

## Run it

```bash
# 1. Python 3.12 (ROCm wheels are 3.12-only; other versions silently fall back to CUDA wheels)
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

# 2. vLLM for ROCm 10.0 from the project's own wheel index — this bundles the ROCm SDK and the gfx1201 kernels
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade

# 3. match torchaudio to the ROCm stack (the CUDA build pulls in and then fails to load _torchaudio.so)
uv pip install "torchaudio==2.11.0+rocm10.0.0" \
  --extra-index-url https://wheels.vllm.ai/rocm/ --index-strategy unsafe-best-match

# 4. vLLM-Omni itself, without letting it re-resolve vLLM
git clone https://github.com/vllm-project/vllm-omni && cd vllm-omni
uv pip install -r requirements/common.txt -r requirements/rocm.txt
uv pip install --no-deps -e .

# 5. weights
export HF_HOME=$PWD/../hf
hf download fishaudio/drama-3-int4

# 6. serve on the second card
export ROCR_VISIBLE_DEVICES=1
export VLLM_DISABLED_KERNELS=RDNAHybridW4A16LinearKernel
vllm serve "$(ls -d $HF_HOME/hub/models--fishaudio--drama-3-int4/snapshots/*/ | head -1)" \
  --omni --port 8091 --gpu-memory-utilization 0.85 --max-num-seqs 4
```

Cold start to `Application startup complete`: **≈ 5 minutes** (weights 22.45 GiB consumed + DAC codec stage + CUDA-graph
capture of 39 piecewise + 3 full graphs). Steady-state VRAM: **29.6 GB of 32 GB** on the serving card.

### One flag at a time

| step | what it buys |
|---|---|
| `VLLM_DISABLED_KERNELS=RDNAHybridW4A16LinearKernel` | unblocks the load (the stride assertion) |
| `ROCR_VISIBLE_DEVICES=1` | puts the model on the free card — setting **both** `ROCR_` and `HIP_` to `1` double-masks and yields "No CUDA GPUs are available" |
| `--gpu-memory-utilization 0.85` | 22.45 GiB weights + activation + graph memory leaves ~3 GiB of KV |
| `--max-num-seqs 4` | matches the KV budget; more sequences than KV is a runtime failure, not a queue |

## Measured (warm server, 3 runs per row, whole-WAV responses)

| input | words | wall (median) | audio | **RTF** | first audio (streamed) |
|---|---:|---:|---:|---:|---:|
| one sentence | 9 | 2.43 s | 3.20 s | **1.32×** | — |
| one paragraph | 43 | 10.26 s | 15.70 s | **1.57×** | — |
| five sentences | 77 | 12.11 s | 23.22 s | **1.92×** | — |
| streamed, one sentence | 9 | 4.00 s total | — | — | **0.287 s** |

RTF is *audio seconds per wall second*, so higher is better: 1.3–1.9× means the card synthesizes faster than playback.
Short inputs are dominated by per-request overhead — the longer the text, the better the ratio gets.

Samples: `bench/audio-r9700/b-{10w,50w,long}-0-16k.wav` (16 kHz downsamples of the actual takes).
Raw numbers: `bench/bench-r9700.json`; harness: `scripts/bench-r9700.py`.

## What it means for live chat

- **First audio 0.29 s streamed** is comfortably inside conversational latency: a reply can start speaking almost
  immediately and keep ahead of playback at 1.3–1.9× realtime for anything past a short sentence.
- **Sustained throughput is ~3.2× lower than the RTX 5090** (5.85–5.98× RTF there, 1.32–1.92× here), so long-form
  narration — a full scene, an audiobook chapter — is the weak case, and multi-character scenes will queue.
- The card is **not** idle: the assistant host's embedding, reranker and clef-flash classifier services hold ~24 GiB of the first
  R9700. This recipe therefore needs the *second* card; the two are indistinguishable in `torch.cuda.get_device_name`
  and differ only by PCI address and unique id.

## Caveats we did not hide

- vLLM-Omni warns that its version does not match vLLM 0.33.0's expected major/minor. It runs, but a vLLM upgrade may
  break it — pin both.
- The model id served is the **snapshot path**, not `fishaudio/drama-3-int4`; a request with the friendly name 404s.
- `--max-num-seqs 4` is what the KV budget allows at 0.85 utilization. Raising it needs a bigger KV slice or a smaller
  stage-0 footprint.
- We measured with whole-WAV and one streaming case. Concurrency, long-form chunking and reference-clip cloning on this
  card are still unmeasured (cloning is supported by the runtime, not yet benchmarked here).
