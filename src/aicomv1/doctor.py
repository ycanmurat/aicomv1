from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess

from aicomv1.api import build_services
from aicomv1.config import Settings


def _memory_gb() -> str:
    if platform.system() != "Darwin":
        return "bilinmiyor"
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return "bilinmiyor"
    return f"{int(result.stdout.strip()) / 1024**3:.0f} GB"


async def diagnose() -> bool:
    settings = Settings.from_env()
    services = build_services(settings)
    llm = await services.llm.status()
    stt, tts = await asyncio.gather(
        asyncio.to_thread(services.stt.status), asyncio.to_thread(services.tts.status)
    )
    print(f"AICOM v1 — {platform.system()} {platform.machine()}, RAM {_memory_gb()}")
    print(f"Python {platform.python_version()}")
    print()
    for status in (llm, stt, tts):
        mark = "OK" if status.ready else "EKSİK"
        print(f"[{mark:5}] {status.name}: {status.detail}")
    print(f"[{'OK' if shutil.which(settings.ffmpeg_executable) else 'EKSİK':5}] ffmpeg")
    print(f"[{'OK' if shutil.which('ollama') else 'EKSİK':5}] ollama komutu")
    print(f"[BİLGİ] Yerel bilgi belgesi: {services.knowledge.count()}")
    print()
    ready = llm.ready and stt.ready and tts.ready
    print("Sistem konuşmaya hazır." if ready else "Eksikleri kurmak için: ./scripts/bootstrap.sh")
    return ready


def main() -> None:
    raise SystemExit(0 if asyncio.run(diagnose()) else 1)


if __name__ == "__main__":
    main()
