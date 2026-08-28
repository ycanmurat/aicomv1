# Architecture

AICOM is a local-first, bilingual voice-assistant pipeline designed primarily for Apple
Silicon Macs. The architecture favors a responsive conversation loop and replaceable local
providers over a single large model.

## Runtime flow

```text
Browser microphone
  -> AudioWorklet (16 kHz mono PCM)
  -> browser VAD and turn endpointing
  -> persistent WebSocket
  -> Whisper or Nemotron speech recognition
  -> session memory, local tools, and SQLite knowledge search
  -> Qwen through the local Ollama API
  -> streaming clause segmenter
  -> FreyaTTS or an offline macOS voice
  -> interruptible browser audio queue
```

The server does not wait for the complete model response before starting speech synthesis.
Text deltas are sent to the interface immediately; each completed clause enters the TTS
queue while the model continues generating the next one.

## Components

| Area | Implementation | Notes |
|---|---|---|
| Web client | HTML, CSS, JavaScript, AudioWorklet | Captures audio, detects speech, plays queued WAV segments, and supports barge-in |
| Transport | FastAPI WebSocket | Carries binary PCM input and JSON response events over one persistent connection |
| STT | whisper.cpp by default | Accuracy-oriented, turn-based transcription with Turkish and English language hints |
| Experimental STT | Nemotron 3.5 through NeMo-Speech.cpp | Lower-latency local path; enabled explicitly after the full setup |
| LLM | Qwen3.5 9B through Ollama | Streams tokens from the configured Ollama endpoint; loopback by default |
| TTS | FreyaTTS and macOS speech | Freya is used when available; English falls back to an offline English macOS voice |
| Memory | In-process sessions + summaries | Keeps recent messages and compacts older context |
| Knowledge | SQLite FTS5 | Searches only documents explicitly added to the local database |
| Tools | Allowlisted local functions | Currently includes deterministic calculation and local time |

## Language handling

Each session carries an explicit `tr` or `en` language setting. It controls the system
prompt, speech-recognition hint, summary language, speech voice, and interface copy. The
language is selected by the user instead of being inferred independently by every provider,
which keeps a turn consistent from transcription through playback.

FreyaTTS is currently the Turkish-oriented voice path. English speech uses the configured
offline macOS voice when Freya is active.

`POST /api/sessions` accepts an optional `{"language":"en"}` or `{"language":"tr"}` body.
The WebSocket announces that language on connection. A `language.set` event updates the
session, cancels the active turn, and discards any partial recording. Text and `audio.start`
events can also carry a language; each turn captures it so later changes cannot alter its
transcript, prompt, or voice. Reconnecting to the same session preserves its language.

## Turn lifecycle and interruption

The browser keeps a small pre-roll buffer so the beginning of a detected utterance is not
lost. After silence reaches the endpoint threshold, it commits the buffered PCM as one turn.
The server then writes a turn WAV file, transcribes it, and starts an assistant task.

Every active turn has a cancellation signal. Speaking while AICOM is playing audio clears
the browser queue, stops the current clip, and cancels further model and TTS output. Some
native provider work may finish in the background because not every local runtime supports
hard cancellation.

## Local data boundary

- The HTTP server and Ollama endpoint use loopback addresses by default.
- Session audio is stored under `data/audio/`.
- The knowledge database is stored under `data/knowledge/`.
- STT and TTS assets are stored under `models/` and `.runtime/`; LLM weights stay in
  Ollama's local model store.
- Model assets, audio, `.env`, and the personal knowledge database are excluded from Git.
- Audio downloads are restricted to the active session directory to prevent path traversal.

Model installation requires network access. Normal conversation does not require a cloud AI
service once all selected models are present locally.

## Development checks

The bootstrap script installs test dependencies. After setup, use the installed environment
without downloading or synchronizing packages:

```bash
uv run --offline --no-sync pytest
uv run --offline --no-sync ruff check .
uv run --offline --no-sync aicom-doctor
uv run --offline --no-sync aicom-benchmark --language en --voice
uv run --offline --no-sync aicom-smoke sample.wav --language tr
```

The smoke test expects a mono, PCM16 WAV at 16 kHz and a running AICOM server. Run it once
with an English sample and once with a Turkish sample. It checks transcription, response
text, and generated WAV output; it is not a microphone or perceptual voice-quality test.

When changing dependencies, use `uv sync --extra dev --extra freya --python 3.11` to retain
the optional Freya environment, or omit `--extra freya` for core-only development.

## Known design limits

The current endpoint detector is energy-based rather than semantic. Whisper returns one
transcript after the user finishes speaking, and clause-level WAV files can introduce small
rhythm breaks. Local model quality is constrained by available memory and compute. These
trade-offs make the project a practical experimental base, not a production-grade realtime
assistant.

The provider boundaries are intentionally small so streaming ASR, semantic turn detection,
continuous PCM synthesis, or a different local model can be added without replacing the
whole application.
