from aicomv1.providers.stt_nemotron import NemotronCppTranscriber
from aicomv1.providers.stt_whisper import _clean_transcript


def test_whisper_transcript_cleanup() -> None:
    assert _clean_transcript("<|tr|>  Merhaba dünya. [Müzik]") == "Merhaba dünya."
    assert _clean_transcript("la la la la la la la la la") == ""


def test_nemotron_json_output_cleanup() -> None:
    output = '{"text":"Merhaba"}\n{"final_text":"dünya"}\n'
    assert NemotronCppTranscriber._extract_text(output) == "Merhaba"


def test_nemotron_pretty_json_output_cleanup() -> None:
    output = """{
      "file": "input.wav",
      "text": "Bugün sesli iletişim deniyoruz.",
      "words": [{"word": "Bugün"},]
    }"""
    assert NemotronCppTranscriber._extract_text(output) == "Bugün sesli iletişim deniyoruz."
