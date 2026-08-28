from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path
from threading import Event

from aicomv1.api import build_services
from aicomv1.config import Settings
from aicomv1.prompt import SYSTEM_PROMPT

DEFAULT_PROMPTS = (
    "Kuantum bilgisayarların klasik bilgisayarlardan farkını kısa ve anlaşılır anlat.",
    "Bir proje gecikiyorsa kök nedeni bulmak için hangi üç soruyla başlamalıyım?",
    "1250 liranın yüzde 18'i kaç eder?",
)


async def run_benchmark(prompts: tuple[str, ...], voice: bool) -> None:
    settings = Settings.from_env()
    services = build_services(settings)
    first_token_values: list[int] = []
    total_values: list[int] = []
    for index, prompt in enumerate(prompts, 1):
        started = time.perf_counter()
        first_token: int | None = None
        chunks: list[str] = []
        async for delta in services.llm.stream(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            cancel=Event(),
        ):
            if first_token is None:
                first_token = round((time.perf_counter() - started) * 1000)
            chunks.append(delta)
        total = round((time.perf_counter() - started) * 1000)
        first_token_values.append(first_token or total)
        total_values.append(total)
        answer = "".join(chunks).strip()
        print(f"{index}. ilk token={first_token or total} ms, toplam={total} ms")
        print(f"   {answer[:180]}")
        if voice and answer:
            with tempfile.TemporaryDirectory(prefix="aicom-bench-") as temp_dir:
                audio = await asyncio.to_thread(
                    services.tts.synthesize,
                    answer[:220],
                    Path(temp_dir) / "speech.wav",
                )
                print(f"   TTS={audio.elapsed_ms} ms ({audio.provider})")
    print()
    print(
        "Ortalama: "
        f"ilk token {statistics.mean(first_token_values):.0f} ms, "
        f"tam yanıt {statistics.mean(total_values):.0f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AICOM yerel gecikme ölçümü")
    parser.add_argument("--prompt", action="append", help="Ölçülecek özel istem")
    parser.add_argument("--voice", action="store_true", help="TTS süresini de ölç")
    args = parser.parse_args()
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
    asyncio.run(run_benchmark(prompts, args.voice))


if __name__ == "__main__":
    main()
