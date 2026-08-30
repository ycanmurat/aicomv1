"""Explicit setup-time downloads only, scoped to the disposable lab directory."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = (
    ("MOSS-TTS-Nano-100M-ONNX", "f52645cb467506d8e18e746ddd59482685b74e58"),
    ("MOSS-Audio-Tokenizer-Nano-ONNX", "ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae"),
)


def main() -> None:
    root = Path(__file__).resolve().parents[2] / ".runtime/voice-lab/moss-tts-nano/models"
    for name, revision in MODELS:
        snapshot_download(
            repo_id=f"OpenMOSS-Team/{name}",
            revision=revision,
            local_dir=root / name,
            allow_patterns=["*.onnx", "*.data", "*.json", "*.model", "README.md"],
        )
    print("MOSS ONNX assets are ready. Runtime inference will not download anything.")


if __name__ == "__main__":
    main()
