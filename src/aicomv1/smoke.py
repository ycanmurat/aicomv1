from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import httpx
import websockets

from aicomv1.prompt import normalize_language


async def run_smoke(audio_path: Path, base_url: str, language: str = "en") -> None:
    language = normalize_language(language)
    http_url = base_url.rstrip("/")
    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{http_url}/api/sessions", json={"language": language})
        response.raise_for_status()
        session_id = response.json()["id"]
        try:
            with wave.open(str(audio_path), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16_000):
                    raise ValueError("The smoke-test WAV must be mono PCM16 at 16 kHz.")
                raw_audio = wav.readframes(wav.getnframes())

            transcript = ""
            answer = ""
            audio_events = 0
            async with websockets.connect(
                f"{ws_url}/api/realtime/{session_id}", max_size=4 * 1024 * 1024
            ) as websocket:
                while True:
                    initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
                    if initial.get("type") == "language":
                        if initial.get("language") != language:
                            raise RuntimeError(f"Unexpected session language: {initial}")
                        continue
                    if initial.get("type") == "state" and initial.get("state") == "listening":
                        break
                    raise RuntimeError(f"Unexpected initial event: {initial}")
                await websocket.send(json.dumps({"type": "audio.start", "language": language}))
                for offset in range(0, len(raw_audio), 6400):
                    await websocket.send(raw_audio[offset : offset + 6400])
                await websocket.send(json.dumps({"type": "audio.commit"}))

                while True:
                    event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=180))
                    event_type = event.get("type")
                    if event_type == "transcript":
                        if event.get("language") != language:
                            raise RuntimeError(
                                "The transcription language does not match the session."
                            )
                        transcript = str(event.get("text", ""))
                        print(
                            f"STT [{event.get('provider')} / {event.get('transcription_ms')} ms]: "
                            f"{transcript}"
                        )
                    elif event_type == "text_delta":
                        answer += str(event.get("delta", ""))
                    elif event_type == "audio":
                        audio_response = await client.get(f"{http_url}{event['url']}")
                        audio_response.raise_for_status()
                        if not audio_response.content.startswith(b"RIFF"):
                            raise RuntimeError("Generated audio is not a valid WAV file.")
                        audio_events += 1
                        print(
                            f"TTS [{event.get('provider')} / {event.get('synthesis_ms')} ms]: "
                            f"{len(audio_response.content)} bytes"
                        )
                    elif event_type == "error":
                        raise RuntimeError(str(event.get("message", "Unknown error")))
                    elif event_type == "warning" and event.get("code") == "speech_not_understood":
                        raise RuntimeError("Speech was not understood; record a clearer sample.")
                    elif event_type == "metrics":
                        print(f"LLM: {answer.strip()}")
                        print(
                            f"Latency: token={event.get('first_token_ms')} ms, "
                            f"voice={event.get('first_audio_ms')} ms, "
                            f"total={event.get('total_ms')} ms"
                        )
                        break
            if not transcript or not answer or audio_events == 0:
                raise RuntimeError("The end-to-end turn did not produce text and audio.")
            print(f"Local end-to-end voice test passed ({language}).")
        finally:
            await client.delete(f"{http_url}/api/sessions/{session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end voice test against a running AICOM server",
        epilog="Examples: aicom-smoke english.wav --language en; "
        "aicom-smoke turkish.wav --language tr",
    )
    parser.add_argument("audio", type=Path, help="Mono PCM16 WAV at 16 kHz")
    parser.add_argument("--url", default="http://127.0.0.1:7870")
    parser.add_argument("--language", choices=("en", "tr"), default="en", help="Recording language")
    args = parser.parse_args()
    asyncio.run(run_smoke(args.audio.resolve(), args.url, args.language))


if __name__ == "__main__":
    main()
