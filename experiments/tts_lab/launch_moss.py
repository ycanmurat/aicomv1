from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano yerel deney ekranı")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    lab_root = project_root / ".runtime" / "voice-lab"
    moss_root = lab_root / "moss-tts-nano"
    sys.path.insert(0, str(moss_root))

    import app as legacy_app
    from app_onnx import (
        OnnxNanoTTSServiceAdapter,
        OnnxRequestRuntimeManager,
        _render_index_html_onnx,
    )

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    output_dir = lab_root / "output"
    runtime = OnnxNanoTTSServiceAdapter(
        model_dir=moss_root / "models",
        output_dir=output_dir,
        cpu_threads=max(1, args.cpu_threads),
        execution_provider="cpu",
        max_new_frames=375,
        text_normalizer_manager=None,
    )
    warmup_manager = legacy_app.WarmupManager(runtime, text_normalizer_manager=None)
    warmup_manager.start()

    OnnxRequestRuntimeManager._factory_model_dir = runtime.model_dir
    OnnxRequestRuntimeManager._factory_output_dir = output_dir
    OnnxRequestRuntimeManager._factory_max_new_frames = 375
    OnnxRequestRuntimeManager._factory_execution_provider = runtime.execution_provider
    OnnxRequestRuntimeManager._factory_text_normalizer_manager = None
    legacy_app.RequestRuntimeManager = OnnxRequestRuntimeManager
    legacy_app._render_index_html = _render_index_html_onnx

    prepare_request_texts = legacy_app.shared_prepare_tts_request_texts

    def prepare_turkish_request_texts(**kwargs):
        kwargs["enable_wetext"] = False
        kwargs["text_normalizer_manager"] = None
        return prepare_request_texts(**kwargs)

    legacy_app.shared_prepare_tts_request_texts = prepare_turkish_request_texts

    application = legacy_app._build_app(runtime, warmup_manager, None, None)
    application.title = "MOSS-TTS-Nano Türkçe Deney Alanı"
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
