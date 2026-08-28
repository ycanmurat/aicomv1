from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import httpx
import websockets


async def run_smoke(audio_path: Path, base_url: str) -> None:
    http_url = base_url.rstrip("/")
    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{http_url}/api/sessions")
        response.raise_for_status()
        session_id = response.json()["id"]
        try:
            with wave.open(str(audio_path), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16_000):
                    raise ValueError("Duman testi WAV dosyası mono, PCM16 ve 16 kHz olmalıdır.")
                raw_audio = wav.readframes(wav.getnframes())

            transcript = ""
            answer = ""
            audio_events = 0
            async with websockets.connect(
                f"{ws_url}/api/realtime/{session_id}", max_size=4 * 1024 * 1024
            ) as websocket:
                initial = json.loads(await websocket.recv())
                if initial.get("state") != "listening":
                    raise RuntimeError(f"Beklenmeyen başlangıç durumu: {initial}")
                await websocket.send(json.dumps({"type": "audio.start"}))
                for offset in range(0, len(raw_audio), 6400):
                    await websocket.send(raw_audio[offset : offset + 6400])
                await websocket.send(json.dumps({"type": "audio.commit"}))

                while True:
                    event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=180))
                    event_type = event.get("type")
                    if event_type == "transcript":
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
                            raise RuntimeError("Üretilen ses geçerli WAV değil.")
                        audio_events += 1
                        print(
                            f"TTS [{event.get('provider')} / {event.get('synthesis_ms')} ms]: "
                            f"{len(audio_response.content)} bayt"
                        )
                    elif event_type == "error":
                        raise RuntimeError(str(event.get("message", "Bilinmeyen hata")))
                    elif event_type == "metrics":
                        print(f"LLM: {answer.strip()}")
                        print(
                            f"Gecikme: token={event.get('first_token_ms')} ms, "
                            f"ses={event.get('first_audio_ms')} ms, "
                            f"toplam={event.get('total_ms')} ms"
                        )
                        break
            if not transcript or not answer or audio_events == 0:
                raise RuntimeError("Uçtan uca tur eksik tamamlandı.")
            print("Uçtan uca yerel ses testi başarılı.")
        finally:
            await client.delete(f"{http_url}/api/sessions/{session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Çalışan AICOM sunucusunda uçtan uca test")
    parser.add_argument("audio", type=Path, help="Mono PCM16 16 kHz WAV")
    parser.add_argument("--url", default="http://127.0.0.1:7870")
    args = parser.parse_args()
    asyncio.run(run_smoke(args.audio.resolve(), args.url))


if __name__ == "__main__":
    main()
