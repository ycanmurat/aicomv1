"""Single-owner adapter around the pinned MOSS ONNX frame callback."""

from __future__ import annotations

import resource
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np

MOSS_REVISION = "cc7bdf19c7639c0870dab22045a33b442760f6be"
FILLER_TEXT = "Bir saniye, bakıyorum."


class GenerationCancelled(Exception):
    """The owning request stopped; no further audio should be emitted."""


def check_cancelled(stop: threading.Event) -> None:
    if stop.is_set():
        raise GenerationCancelled


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024**2 if sys.platform == "darwin" else 1024), 1)


def prepare_text(text: str) -> str:
    """Preserve Turkish letters and raw numbers; never use en/zh normalization."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c.isspace())
    return " ".join(text.split())


def load_runtime(lab_root: Path, threads: int):
    moss_root = lab_root / "moss-tts-nano"
    if not (moss_root / "ort_cpu_runtime.py").is_file():
        raise RuntimeError("Önce experiments/tts_lab/setup_moss.sh komutunu çalıştırın.")
    sys.path.insert(0, str(moss_root))
    import onnxruntime as ort
    from onnx_tts_runtime import OnnxTtsRuntime

    class StreamingRuntime(OnnxTtsRuntime):
        def _session(self, path_value):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = self.thread_count
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            return ort.InferenceSession(
                str(path_value), sess_options=options, providers=["CPUExecutionProvider"]
            )

        def _create_sessions(self):
            # The same fixed-sampling graph as upstream, without unused full-wave
            # or alternative sampling graphs that duplicate weights in memory.
            tts = self.tts_meta_path.parent
            codec = self.codec_meta_path.parent
            return {
                "prefill": self._session(tts / self.tts_meta["files"]["prefill"]),
                "decode": self._session(tts / self.tts_meta["files"]["decode_step"]),
                "local_fixed_sampled_frame": self._session(
                    tts / self.tts_meta["files"]["local_fixed_sampled_frame"]
                ),
                "codec_encode": self._session(codec / self.codec_meta["files"]["encode"]),
                "codec_decode_step": self._session(codec / self.codec_meta["files"]["decode_step"]),
            }

    return StreamingRuntime(
        model_dir=moss_root / "models",
        thread_count=threads,
        max_new_frames=375,
        sample_mode="fixed",
        do_sample=True,
        execution_provider="cpu",
        output_dir=lab_root / "output",
    )


class MossEngine:
    """Call from one worker only; the server owns the concurrency lock."""

    def __init__(self, lab_root: Path, reference: Path | None = None, threads: int = 4):
        self.lab_root = lab_root
        self.reference = reference
        self.threads = threads
        self.runtime = None
        self.prompt_codes = None
        self.reference_key = None
        self.filler: list[np.ndarray] = []
        self.load_seconds = 0.0

    @property
    def reference_name(self) -> str:
        return self.reference.name if self.reference else "Bella (yerleşik referans)"

    @property
    def sample_rate(self) -> int:
        return int(self.runtime.codec_meta["codec_config"]["sample_rate"])

    def load(self) -> None:
        if self.runtime is None:
            started = time.perf_counter()
            self.runtime = load_runtime(self.lab_root, self.threads)
            self.load_seconds = time.perf_counter() - started

    def set_reference(self, reference: Path) -> None:
        self.reference = reference
        self.prompt_codes = None
        self.reference_key = None
        self.filler.clear()

    def _get_prompt_codes(self):
        key = None
        if self.reference:
            stat = self.reference.stat()
            key = (str(self.reference.resolve()), stat.st_mtime_ns, stat.st_size)
        if self.prompt_codes is None or key != self.reference_key:
            self.prompt_codes = self.runtime.resolve_prompt_audio_codes(
                voice="Bella", prompt_audio_path=self.reference
            )
            self.reference_key = key
            self.filler.clear()
        return self.prompt_codes

    def warmup(self, stop: threading.Event | None = None) -> None:
        self.load()
        chunks = []
        self.synthesize(
            FILLER_TEXT, lambda pcm, _meta: chunks.append(pcm.copy()), stop or threading.Event()
        )
        self.filler = chunks

    def synthesize(
        self,
        text: str,
        on_audio: Callable[[np.ndarray, dict], None],
        stop: threading.Event,
        *,
        seed: int = 42,
        max_frames: int = 375,
    ) -> dict:
        check_cancelled(stop)
        self.load()
        started = time.perf_counter()
        runtime = self.runtime
        text = prepare_text(text)
        if not text:
            raise ValueError("Seslendirilecek metin boş olamaz.")
        prompt_codes = self._get_prompt_codes()
        check_cancelled(stop)
        runtime.rng = np.random.default_rng(seed)
        runtime.manifest["generation_defaults"]["max_new_frames"] = max_frames
        text_chunks = runtime.split_voice_clone_text(text, max_tokens=75)
        sample_rate = self.sample_rate
        total_samples = 0
        chunk_count = 0
        first_pcm = None
        frame_limit_hit = False
        previous_emitted_at = None
        max_gap_ms = 0.0
        minimum_lead = float("inf")
        from ort_cpu_runtime import _resolve_stream_decode_frame_budget

        for text_index, chunk_text in enumerate(text_chunks):
            check_cancelled(stop)
            rows = runtime.build_voice_clone_request_rows(
                prompt_codes, runtime.encode_text(chunk_text)
            )
            pending = []
            runtime.codec_streaming_session.reset()

            def decode_pending(force=False, pending=pending, text_index=text_index):
                nonlocal first_pcm, total_samples, chunk_count, previous_emitted_at
                nonlocal max_gap_ms, minimum_lead
                check_cancelled(stop)
                budget = _resolve_stream_decode_frame_budget(total_samples, sample_rate, first_pcm)
                if not pending or (not force and len(pending) < budget):
                    return
                frames = pending[: len(pending) if force else budget]
                del pending[: len(frames)]
                decoded = runtime.codec_streaming_session.run_frames(frames)
                check_cancelled(stop)
                if decoded is None or decoded[1] <= 0:
                    return
                audio, length = decoded
                waveform = np.asarray(audio[0, :, :length].T, dtype=np.float32)
                if not np.isfinite(waveform).all():
                    raise RuntimeError("Model geçersiz ses örnekleri üretti.")
                now = time.perf_counter()
                if first_pcm is None:
                    first_pcm = now
                if previous_emitted_at is not None:
                    max_gap_ms = max(max_gap_ms, (now - previous_emitted_at) * 1000)
                    minimum_lead = min(
                        minimum_lead, total_samples / sample_rate - (now - first_pcm)
                    )
                previous_emitted_at = now
                total_samples += length
                chunk_count += 1
                pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2")
                on_audio(pcm, {"sample_rate": sample_rate, "text_chunk": text_index})

            def on_frame(_frames, _step_index, frame, pending=pending, decode=decode_pending):
                check_cancelled(stop)
                pending.append(list(frame))
                decode()

            try:
                frames = runtime.generate_audio_frames(rows, on_frame=on_frame)
                frame_limit_hit |= len(frames) >= max_frames
                decode_pending(force=True)
            finally:
                runtime.codec_streaming_session.reset()

        elapsed = time.perf_counter() - started
        if first_pcm is None:
            raise RuntimeError(
                "MOSS bu metin için ses üretmedi; başka bir metin veya referans deneyin."
            )
        audio_seconds = total_samples / sample_rate
        return {
            "first_pcm_ms": round((first_pcm - started) * 1000, 1),
            "generation_seconds": round(elapsed, 3),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(elapsed / audio_seconds, 3),
            "pcm_chunks": chunk_count,
            "text_chunks": text_chunks,
            "max_chunk_gap_ms": round(max_gap_ms, 1),
            "minimum_buffer_lead_ms": round(minimum_lead * 1000, 1) if chunk_count > 1 else None,
            "frame_limit_hit": frame_limit_hit,
            "process_peak_rss_mb": peak_rss_mb(),
            "cpu_threads": self.threads,
        }
