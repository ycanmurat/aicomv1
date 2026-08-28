from __future__ import annotations

import argparse
import asyncio
import platform
import shutil
import subprocess

from aicomv1.api import build_services
from aicomv1.config import Settings
from aicomv1.prompt import normalize_language


def _memory_gb() -> str:
    if platform.system() != "Darwin":
        return "unknown"
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return "unknown"
    return f"{int(result.stdout.strip()) / 1024**3:.0f} GB"


async def diagnose(language: str | None = None) -> bool:
    settings = Settings.from_env()
    services = build_services(settings)
    languages = (normalize_language(language),) if language is not None else ("en", "tr")
    llm = await services.llm.status()
    stt = await asyncio.to_thread(services.stt.status)
    voices = await asyncio.gather(
        *(asyncio.to_thread(services.tts.status, language=code) for code in languages)
    )
    print(f"AICOM v1 — {platform.system()} {platform.machine()}, RAM {_memory_gb()}")
    print(f"Python {platform.python_version()}")
    print()
    for status in (llm, stt):
        mark = "OK" if status.ready else "MISSING"
        print(f"[{mark:5}] {status.name}: {status.detail}")
    for code, status in zip(languages, voices, strict=True):
        mark = "OK" if status.ready else "MISSING"
        print(f"[{mark:5}] {status.name} ({code}): {status.detail}")
    print(f"[{'OK' if shutil.which(settings.ffmpeg_executable) else 'MISSING':5}] ffmpeg")
    print(f"[{'OK' if shutil.which('ollama') else 'MISSING':5}] ollama command")
    print(f"[INFO] Local knowledge documents: {services.knowledge.count()}")
    print()
    ready = llm.ready and stt.ready and all(voice.ready for voice in voices)
    print(
        "Ready for conversation." if ready else "Install missing components: ./scripts/bootstrap.sh"
    )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local AICOM models and voice availability")
    parser.add_argument(
        "--language", choices=("en", "tr"), help="Check one voice language (default: both)"
    )
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(diagnose(args.language)) else 1)


if __name__ == "__main__":
    main()
