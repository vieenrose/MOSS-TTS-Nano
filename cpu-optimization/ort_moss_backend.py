"""MOSS-TTS-Nano-100M via onnxruntime (reference ONNX pipeline) — for the A/B demo.

Same interface as the rs.cpp MossBackend (available / set_reference / synth) but the
AR generation + codec run through onnxruntime (NanoRuntime / OrtCpuRuntime) instead of
ggml. Shares the PrimeTTS v2 frontend (text_norm) + the zh-TW default voice.
"""
import os
import numpy as np

SR_OUT = 48000

class OrtMossBackend:
    def __init__(self, onnx_ar, codec_encode_onnx, threads=2):
        self.available = False
        self.ref_codes = {}
        try:
            from bench_nano_cpu import NanoRuntime
            import onnxruntime as ort
            self.rt = NanoRuntime(onnx_ar, thread_count=threads)
            so = ort.SessionOptions(); so.intra_op_num_threads = threads
            self.enc = ort.InferenceSession(codec_encode_onnx, so, providers=["CPUExecutionProvider"])
            try:
                import text_norm; self._tn = text_norm
            except Exception:
                self._tn = None
            man = self.rt.manifest
            self.tc = man["tts_config"]; self.NVQ = int(self.tc["n_vq"]); self.PAD = int(self.tc["audio_pad_token_id"])
            bv = man.get("builtin_voices") or []
            pick = next((v for v in bv if str(v.get("audio_file", "")).startswith("zh_4")
                         or v.get("voice") == "Yuewen"), None) or (bv[0] if bv else None)
            self.default_ref = pick["prompt_audio_codes"] if pick else None
            self.available = True
            print("[ort-moss] backend ready (onnxruntime AR + codec)")
        except Exception as e:
            print(f"[ort-moss] unavailable ({e})")

    def _normalize(self, t):
        try:
            return self._tn.normalize(t) if self._tn else t
        except Exception:
            return t

    def set_reference(self, session, wav48k):
        a = np.stack([wav48k, wav48k])[None].astype(np.float32)   # stereo
        codes = self.enc.run(None, {"waveform": a, "input_lengths": np.array([a.shape[-1]], np.int32)})[0]
        self.ref_codes[session] = codes[0].astype(np.int32).tolist()

    def clear_reference(self, session):
        self.ref_codes.pop(session, None)

    def synth(self, text, session=None, seed=7, max_frames=400):
        import ort_cpu_runtime as O
        ids = self.rt.encode_text(self._normalize(text))
        ref = self.ref_codes.get(session) or self.default_ref or [[self.PAD] * self.NVQ]
        gd = self.rt.manifest["generation_defaults"]
        gd["do_sample"] = True; gd["sample_mode"] = O.SAMPLE_MODE_FIXED; gd["max_new_frames"] = max_frames
        rows = self.rt.build_voice_clone_request_rows(ref, ids)
        frames = []
        self.rt.generate_audio_frames(rows, on_frame=lambda gf, i, fr: frames.append(list(fr)))
        if not frames:
            return np.zeros(1, np.float32), SR_OUT
        audio = self.rt.decode_full(frames)         # non-streaming, clean per-utterance decode
        a = np.asarray(audio, np.float32)
        if a.ndim > 1:
            a = a.mean(0) if a.shape[0] <= 2 else a.mean(1)
        return a.reshape(-1), SR_OUT
