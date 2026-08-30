from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_TEXTS = (
    "Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?",
    "Elbette, bir saniye kontrol ediyorum.",
    "Randevunuz yirmi dokuz Ağustos Cumartesi günü saat on dört otuz için oluşturuldu.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano gerçek streaming ölçümü")
    parser.add_argument("--reference-audio", required=True, type=Path)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    lab_root = project_root / ".runtime" / "voice-lab"
    moss_root = lab_root / "moss-tts-nano"
    model_root = moss_root / "models"

    if not args.reference_audio.is_file():
        raise SystemExit(f"Referans ses bulunamadı: {args.reference_audio}")
    if not (moss_root / "app_onnx.py").is_file():
        raise SystemExit("Önce experiments/tts_lab/setup_moss.sh çalıştırılmalı.")

    sys.path.insert(0, str(moss_root))
    from app_onnx import OnnxNanoTTSServiceAdapter

    load_started = time.perf_counter()
    service = OnnxNanoTTSServiceAdapter(
        model_dir=model_root,
        output_dir=lab_root / "output",
        cpu_threads=max(1, args.cpu_threads),
        execution_provider="cpu",
        max_new_frames=375,
    )
    print(
        json.dumps(
            {"olay": "model_yuklendi", "sure_s": round(time.perf_counter() - load_started, 3)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    for case_number, text in enumerate(args.texts or DEFAULT_TEXTS, start=1):
        started = time.perf_counter()
        first_audio_seconds: float | None = None
        audio_chunks = 0
        audio_seconds = 0.0
        engine_seconds: float | None = None

        for event in service.synthesize_stream(
            text=text,
            mode="voice_clone",
            voice=None,
            prompt_audio_path=str(args.reference_audio.resolve()),
            max_new_frames=375,
            voice_clone_max_text_tokens=75,
            attn_implementation="fixed",
            do_sample=True,
            seed=args.seed,
        ):
            now = time.perf_counter()
            if event["type"] == "audio":
                audio_chunks += 1
                if first_audio_seconds is None:
                    first_audio_seconds = now - started
                audio_seconds = float(event["emitted_audio_seconds"])
            elif event["type"] == "result":
                engine_seconds = float(event["elapsed_seconds"])

        wall_seconds = time.perf_counter() - started
        result = {
            "olay": "sonuc",
            "deney": case_number,
            "metin": text,
            "ilk_pcm_s": round(first_audio_seconds or -1.0, 3),
            "toplam_s": round(wall_seconds, 3),
            "ses_s": round(audio_seconds, 3),
            "rtf": round(wall_seconds / audio_seconds, 3) if audio_seconds else None,
            "parca_sayisi": audio_chunks,
            "motor_s": round(engine_seconds, 3) if engine_seconds is not None else None,
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
