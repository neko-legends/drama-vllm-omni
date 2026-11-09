# Sources and provenance

| what | where | whose |
|---|---|---|
| Drama 3 engine + NVFP4 recipe (GB10, Apple silicon) | <https://github.com/plotarmordev/drama-tensorfold> | upstream (`plotarmordev`) |
| GB10 measurements (RTF 0.825, first audio 1.955 s, 24 texts, salt 0) | <https://github.com/plotarmordev/drama-tensorfold/tree/main/results/full-vs-quant-01> — `tables.md`, `card-section.md` | upstream |
| NVFP4 pruned weights (7,369 / 10,240 routed experts) | <https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned> | upstream |
| The model | <https://huggingface.co/fishaudio/drama-3> | Fish Audio (FACL v1.0 weights) |
| INT4 checkpoint used on ROCm | <https://huggingface.co/fishaudio/drama-3-int4> | Fish Audio |
| CUDA desktop build (RTX 5090) | [`INSTALL-CUDA.md`](INSTALL-CUDA.md), [`BENCH-5090.md`](BENCH-5090.md) | this repo |
| ROCm path (Radeon R9700, vLLM-Omni) | [`BENCH-R9700.md`](BENCH-R9700.md) | this repo |
| vLLM-Omni Drama 3 runtime | <https://github.com/vllm-project/vllm-omni> | vLLM project |

## The three RTF figures are not comparable cell-for-cell

| column | protocol |
|---|---|
| DGX Spark GB10 | upstream run `full-vs-quant-01`, NVFP4 arm, reference voice, γ 1.5, 24 English texts, seed salt 0, whole-WAV replies for RTF and streamed replies for first audio, nearest-rank p50 |
| RTX 5090 | this repo, 3 texts (9 / 43 / 77 words), 3 runs each, whole-WAV, median of the three |
| Radeon R9700 | this repo, the same 3 texts and 3 runs, whole-WAV, median of the three |

Real-time factor is **client end-to-end wall time divided by generated audio time**, so lower is faster.
Everything is greedy except where a checkpoint's own defaults say otherwise.
