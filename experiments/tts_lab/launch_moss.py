from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fatma — yerel Türkçe ses deneyi")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--reference-audio", type=Path, default=os.getenv("FATMA_REFERENCE_AUDIO"))
    parser.add_argument("--llm-model", default=os.getenv("FATMA_LLM_MODEL", "qwen3.5:2b-q4_K_M"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from experiments.tts_lab.server import create_app

    application = create_app(
        project_root,
        reference=args.reference_audio,
        cpu_threads=args.cpu_threads,
        llm_model=args.llm_model,
    )
    uvicorn.run(application, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
