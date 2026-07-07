#!/usr/bin/env python3
"""Dynamic int8 quantization of the MOSS-TTS-Nano ONNX AR graphs for faster CPU inference.

The exported ONNX shares weight blobs across graphs (prefill + decode_step both reference
`moss_tts_global_shared.data`; the local graphs share `moss_tts_local_shared.data`). Running
`quantize_dynamic` directly on a graph that references shared *external* data produces a
weightless output. The fix is to load each graph WITH its external data merged, save a
self-contained fp32 model, then quantize that.

Weights-only dynamic QInt8 (activations stay fp32) — near-lossless for this codec-token AR TTS.

Result (measured, 2 CPU threads):
  fp32  RTF 0.31   |   int8  RTF 0.21   (~1.5x faster)   ·   weights 420MB -> 105MB per big graph

Usage:
  python quantize_onnx_int8.py <src_onnx_dir> <dst_int8_dir>
"""
import os, sys, shutil
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

AR_GRAPHS = [
    "moss_tts_prefill.onnx",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_cached_step.onnx",
    "moss_tts_local_decoder.onnx",
    "moss_tts_local_fixed_sampled_frame.onnx",
]


def quantize_graph(src_path: str, dst_path: str, tmp_dir: str = "/tmp") -> tuple[int, int]:
    model = onnx.load(src_path, load_external_data=True)      # pull shared weights in
    tmp = os.path.join(tmp_dir, os.path.basename(src_path))
    onnx.save(model, tmp)                                     # self-contained fp32
    quantize_dynamic(tmp, dst_path, weight_type=QuantType.QInt8)
    os.remove(tmp)
    return os.path.getsize(src_path), os.path.getsize(dst_path)


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for name in AR_GRAPHS:
        sp = os.path.join(src, name)
        if not os.path.exists(sp):
            print(f"skip (missing): {name}"); continue
        _, i8 = quantize_graph(sp, os.path.join(dst, name))
        print(f"quantized {name}: -> {i8 // (1024 * 1024)} MB int8")
    # copy config/tokenizer/meta (everything that is not an .onnx graph or its .data)
    for f in os.listdir(src):
        sp = os.path.join(src, f)
        if os.path.isfile(sp) and not f.endswith(".onnx") and not f.endswith(".data"):
            shutil.copy(sp, os.path.join(dst, f))
    print(f"done -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
