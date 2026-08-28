# AICOM

**A local-first voice companion for Turkish and English.**

AICOM listens through the browser, transcribes speech locally, and starts speaking while a
local language model is still generating its answer. It is built for general conversation,
not scripted call-center flows.

> AICOM is an experimental project. It is useful as a local voice-assistant foundation,
> but it does not match frontier cloud models or production voice systems yet.

## What it can do

- Turkish and English speech, responses, and interface
- Streaming text and clause-by-clause speech, with voice interruption
- Session memory, local knowledge search, a calculator, and local time
- Offline conversations after the initial model setup

## Stack

| Component | Technology |
|---|---|
| Speech recognition | whisper.cpp; optional Nemotron 3.5 ASR |
| Language model | Qwen3.5 9B through Ollama |
| Speech synthesis | FreyaTTS for Turkish; offline macOS voices for both languages |
| Application | FastAPI, WebSocket, vanilla JavaScript, SQLite FTS5 |

The project is currently optimized for Apple Silicon Macs with 16 GB or more of unified
memory. Other systems may require provider or model changes.

## Quick start

Requires macOS on Apple Silicon, `uv`, `ffmpeg`, Ollama, and `whisper-cli`. Python 3.11 is
managed automatically by `uv`. With Homebrew:

```bash
brew install uv ffmpeg ollama whisper-cpp
```

Start the Ollama app, or run `ollama serve` in a separate terminal. Then:

```bash
git clone https://github.com/ycanmurat/aicomv1.git
cd aicomv1
./scripts/bootstrap.sh core
./scripts/run.sh
```

Open [http://127.0.0.1:7870](http://127.0.0.1:7870), select a language, and enable the
microphone.

For the optional full voice stack, including NeMo-Speech.cpp, Nemotron, and FreyaTTS:

```bash
./scripts/bootstrap.sh full
```

The full setup downloads several additional gigabytes. Core mode uses Whisper and offline
macOS speech. Configure providers and voices in `.env`; see [.env.example](.env.example).
List installed voices with `say -v '?'`. If needed, set `AICOM_TTS_VOICE` for Turkish and
`AICOM_TTS_VOICE_EN` for English to voices available on your Mac.

The defaults are tuned for a 16 GB Mac: a 6144-token context, three-minute LLM residency,
two-minute Freya residency, and no startup warmup. This keeps answer quality at the 9B model
level while returning memory after idle periods. The first turn after an unload takes longer;
all limits can be adjusted in `.env`. Freya releases its model references and runtime caches
on a best-effort basis; the operating system may retain some allocator memory for reuse.

## Run and test

```bash
./scripts/run.sh
uv run --offline --no-sync aicom-doctor
uv run --offline --no-sync pytest
uv run --offline --no-sync ruff check .
```

## Current limitations

- A local 9B model cannot provide the reasoning depth or factual coverage of leading cloud
  models. AICOM may still give incomplete or incorrect answers.
- Speech detection is energy-based, and Whisper transcribes after a turn ends. Pauses,
  noisy rooms, and clause-based playback can make conversation feel uneven.
- Voice quality and latency depend on the provider and hardware.
- Current information is unavailable unless it has been added to the local knowledge base.

## Privacy

With the default configuration, speech and language inference run on your computer. Setup
downloads dependencies and models; the launcher uses installed assets without syncing or
downloading them at runtime. Keep `AICOM_OLLAMA_URL` pointed at your local Ollama instance
and use a locally installed model.

Audio, knowledge, and model files are excluded from Git. The server listens on `127.0.0.1`
by default and has no authentication: do not expose it to the internet.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the technical design.

## License

AICOM is released under the [MIT License](LICENSE). Downloaded models and third-party
runtimes retain their own licenses.
