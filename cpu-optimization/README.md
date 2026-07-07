# MOSS-TTS-Nano — CPU / on-device optimization

Notes + tooling for running MOSS-TTS-Nano-100M efficiently off-GPU, from a port effort
across **ONNX Runtime (x86 CPU)** and **ggml (Jetson Nano GPU + CPU)**.

## TL;DR — pick the runtime by hardware
| hardware | fastest runtime | why |
|---|---|---|
| **x86 CPU** (e.g. HF Spaces free tier) | **ONNX Runtime** | ORT **MLAS** kernels (AVX2/512) + persistent model load + `local_cached_step` (KV-cached local) |
| **Jetson Nano GPU** (Maxwell sm_53) | **ggml** (custom warp-reduce matvec) | bandwidth-bound batch-1 matvec; a hand-tuned kernel beats ORT (no ORT-CUDA on this device) |

Measured, same phone-attendant sentence, 2 CPU threads:

| runtime | RTF | note |
|---|---|---|
| ONNX Runtime, fp32 | 0.31 | reference |
| **ONNX Runtime, int8** | **0.21** | **~1.5×**, weights-only dynamic QInt8, near-lossless |
| ggml (gguf Q8), per-call | ~0.5–3× slower on x86 | gguf reload per call + ggml CPU matmul < MLAS |

On the Jetson Nano **GPU**, the ggml path hit **RTF ~0.35** with a custom sm_53 matvec kernel
(2.6× over stock ggml). So the winner flips by device: **ggml on the Nano GPU, ORT on x86 CPU.**

## int8 quantization (the main CPU win)
`quantize_onnx_int8.py` — dynamic QInt8 of the AR graphs. The gotcha: the ONNX export **shares
weight blobs across graphs** (prefill + decode_step share `moss_tts_global_shared.data`).
`quantize_dynamic` on a graph that references shared *external* data yields a weightless output;
the fix is to **load each graph with its external data merged first**, then quantize.

```bash
python cpu-optimization/quantize_onnx_int8.py <onnx_dir> <int8_out_dir>
```

Weights-only dynamic quant keeps activations fp32 → minimal quality impact for this
codec-token AR TTS. ~1.5× faster, ~4× smaller weights per big graph (420 MB → 105 MB).

## Other CPU levers (already in the reference runtime)
- **`local_cached_step`** — KV-cached local decoder: the 16-codebook inner loop runs O(L) instead
  of re-attending the whole growing sequence O(L²). Use `sample_mode="fixed"` (the default).
- **Persistent model load** — load the ONNX sessions once; do not reload per utterance.
- **Threading** — `SessionOptions.intra_op_num_threads`; on a 2-vCPU box, 2 is the sweet spot.
- **Text frontend matters** — for entity-heavy text (phone/extension/serial/email/price), an
  entity-aware normalizer (digit-by-digit vs cardinal by context) markedly beats a plain number
  normalizer (e.g. extension `2580` → `二五八零`, not the cardinal `两千五百八十`).

## Real-time caveat (x86 CPU)
MOSS-Nano-100M is autoregressive: even int8 on 2 shared vCPUs is **RTF > 1** in the wild
(~2 with int8), i.e. a **batch "generate-then-play"** experience, not real-time streaming.
Real-time needs a GPU (or more/faster CPU cores). int8 narrows the gap; it doesn't cross it.

## Related
- ggml port (Jetson Nano, custom Maxwell matvec kernel): `RapidSpeech.cpp` fork, arch `moss_tts_nano`.
- A/B demo (ONNX vs ggml, voice clone, entity frontend): HF Space `Luigi/PrimeTTS-vs-Inflect-Nano-v1`.
