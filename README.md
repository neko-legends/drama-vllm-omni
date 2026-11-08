# drama-vllm-omni

Fish Audio Drama 3 text-to-speech on cards like the Radeon 9700 pro.

Measured on one RTX 5090 (sm_120) with TensorFold's native NVFP4 engine:

- [`docs/BENCH-5090.md`](docs/BENCH-5090.md) — warm-server benchmark: latency, real-time
  factor (~5.8×), streamed first-chunk latency (~0.07 s), speaker tags, VRAM, curl repro,
  control-script transcript.
- [`docs/INSTALL-RIPPER.md`](docs/INSTALL-RIPPER.md) — fresh Omarchy machine to a serving
  engine: pacman names, CUDA/Zig versions, build commands, weights, serve flags, gotchas.
- `scripts/drama3-service.sh` — idempotent `start|stop|status|restart` for the engine on
  `ripper`, GPU-locked, safe to drive from a UI toggle.
- `scripts/bench-drama3.py`, `scripts/check-speaker-tags.py` — the stdlib harnesses that
  produced the numbers (they start and stop nothing).
- `bench/` — the raw run log (`bench.json`, `speakers.json`) and three audio takes under 1 MB.
