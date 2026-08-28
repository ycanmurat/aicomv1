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
from aicomv1.prompt import normalize_language, system_prompt

DEFAULT_PROMPTS = {
    "en": (
        "Briefly explain how quantum computers differ from classical computers.",
        "Which three questions help identify the root cause of a delayed project?",
        "What is 18 percent of 1250?",
    ),
    "tr": (
        "Kuantum bilgisayarların klasik bilgisayarlardan farkını kısa ve anlaşılır anlat.",
        "Bir proje gecikiyorsa kök nedeni bulmak için hangi üç soruyla başlamalıyım?",
        "1250 liranın yüzde 18'i kaç eder?",
    ),
}


async def run_benchmark(prompts: tuple[str, ...], voice: bool, language: str = "en") -> None:
    language = normalize_language(language)
    if not prompts:
        raise ValueError("At least one benchmark prompt is required.")
    settings = Settings.from_env()
    services = build_services(settings)
    if voice:
        voice_status = await asyncio.to_thread(services.tts.status, language=language)
        if not voice_status.ready:
            raise RuntimeError(voice_status.detail)
        if voice_status.name == "tts-none":
            print("Speech synthesis is disabled; measuring text latency only.")
            voice = False
    print(f"AICOM benchmark — language: {language}")
    first_token_values: list[int] = []
    total_values: list[int] = []
    for index, prompt in enumerate(prompts, 1):
        started = time.perf_counter()
        first_token: int | None = None
        chunks: list[str] = []
        async for delta in services.llm.stream(
            system_prompt=system_prompt(language),
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
        print(f"{index}. first token={first_token or total} ms, total={total} ms")
        print(f"   {answer[:180]}")
        if voice and answer:
            with tempfile.TemporaryDirectory(prefix="aicom-bench-") as temp_dir:
                audio = await asyncio.to_thread(
                    services.tts.synthesize,
                    answer[:220],
                    Path(temp_dir) / "speech.wav",
                    language=language,
                )
                print(f"   TTS={audio.elapsed_ms} ms ({audio.provider})")
    print()
    print(
        "Average: "
        f"first token {statistics.mean(first_token_values):.0f} ms, "
        f"full response {statistics.mean(total_values):.0f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local AICOM response latency")
    parser.add_argument("--prompt", action="append", help="Custom benchmark prompt (repeatable)")
    parser.add_argument("--voice", action="store_true", help="Also measure speech synthesis")
    parser.add_argument(
        "--language", choices=("en", "tr"), default="en", help="Conversation language"
    )
    args = parser.parse_args()
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS[args.language]
    asyncio.run(run_benchmark(prompts, args.voice, args.language))


if __name__ == "__main__":
    main()
