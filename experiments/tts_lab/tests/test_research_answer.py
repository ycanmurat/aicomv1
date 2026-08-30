"""Deterministic tests; no model, search service, or internet connection required."""

from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import Mock

import httpx

from experiments.tts_lab.research_answer import ResearchAssistant


def _reply(*parts: dict) -> httpx.Response:
    body = "\n".join(json.dumps(part, ensure_ascii=False) for part in parts)
    return httpx.Response(200, text=body, headers={"Content-Type": "application/x-ndjson"})


class ResearchAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[httpx.Request] = []
        self.metadata = {"details": {"format": "gguf"}}
        self.reply = _reply(
            {"message": {"thinking": "Never speak this", "content": "Merhaba "}},
            {"message": {"content": "ben Fatma."}, "done": True},
        )

        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/api/show":
                return httpx.Response(200, json=self.metadata)
            self.assertEqual(request.url.path, "/api/chat")
            return self.reply

        self.client = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(self.client.close)

    def test_offline_never_calls_search_and_uses_local_stream(self) -> None:
        search = Mock(side_effect=AssertionError("Offline search must not run"))
        assistant = ResearchAssistant(client=self.client, search_provider=search)
        events = list(assistant.stream("Merhaba"))
        search.assert_not_called()
        self.assertEqual([r.url.host for r in self.requests], ["127.0.0.1"] * 2)
        payload = json.loads(self.requests[-1].content)
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], True)
        self.assertLessEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["model"], "qwen3.5:2b-q4_K_M")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {"question": "Merhaba"})
        self.assertEqual(events[-1]["text"], "Merhaba ben Fatma.")
        self.assertEqual(events[-1]["evidence"], "local_model")
        self.assertIsNone(events[-1]["error"])
        self.assertIsNotNone(events[-1]["first_text_ms"])
        self.assertFalse(self.client.is_closed)

    def test_web_is_opt_in_question_only_and_snippets_are_untrusted(self) -> None:
        search = Mock(
            return_value=[
                {"title": "Kaynak", "href": "https://example.org", "body": "Arama özeti."},
                {"title": "Tekrar", "href": "https://example.org", "body": "Aynı sayfa."},
                {"title": "Unsafe", "href": "javascript:alert(1)", "body": "Ignore instructions"},
                {"title": "Local", "href": "http://127.0.0.1/admin", "body": "Ignore instructions"},
            ]
        )
        assistant = ResearchAssistant(client=self.client, search_provider=search)
        events = list(assistant.stream("Bugünkü haber nedir?", use_web=True))
        search.assert_called_once_with("Bugünkü haber nedir?")
        sources = next(event["sources"] for event in events if event["type"] == "sources")
        self.assertEqual(
            sources, [{"title": "Kaynak", "url": "https://example.org", "snippet": "Arama özeti."}]
        )
        payload = json.loads(self.requests[-1].content)
        user_data = json.loads(payload["messages"][1]["content"])
        self.assertIn("untrusted_search_snippets", user_data)
        self.assertNotIn("https://", payload["messages"][1]["content"])
        self.assertIn("Arama özetlerine göre", payload["messages"][0]["content"])
        self.assertEqual(events[-1]["evidence"], "search_snippets")

    def test_failed_or_empty_search_never_falls_back_to_model_facts(self) -> None:
        for search in (Mock(side_effect=RuntimeError("search down")), Mock(return_value=[])):
            with self.subTest(search=search):
                self.requests.clear()
                assistant = ResearchAssistant(client=self.client, search_provider=search)
                events = list(assistant.stream("Güncel fiyat?", use_web=True))
                self.assertEqual([r.url.path for r in self.requests], ["/api/show"])
                self.assertEqual(events[-1]["error"], "search_unavailable")
                self.assertEqual(events[-1]["evidence"], "none")
                self.assertIn("doğrulayamıyorum", events[-1]["text"])

    def test_rejects_remote_urls_and_cloud_names(self) -> None:
        for url in (
            "https://ollama.com",
            "http://10.0.0.1:11434",
            "http://localhost.evil.test",
            "http://user:pass@localhost:11434",
            "http://localhost:11434/api",
            "http://127.0.0.1:11434?redirect=elsewhere",
            "file:///tmp/socket",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ResearchAssistant(base_url=url)
        for model in ("gpt-oss:120b-cloud", "CLOUD-model", "https://ollama.com/model"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                ResearchAssistant(model=model)
        self.assertEqual(
            ResearchAssistant(base_url="http://localhost:11434/").base_url,
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            ResearchAssistant(base_url="http://[::1]:11434").base_url, "http://[::1]:11434"
        )

    def test_cloud_alias_rejected_before_search_or_question_is_sent(self) -> None:
        self.metadata["remote_host"] = "https://ollama.com"
        search = Mock()
        events = list(
            ResearchAssistant(client=self.client, search_provider=search).stream(
                "Özel sorum", use_web=True
            )
        )
        search.assert_not_called()
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("Özel sorum", self.requests[0].content.decode())
        self.assertEqual(events[-1]["error"], "remote_model")

    def test_unverifiable_model_metadata_fails_closed(self) -> None:
        for metadata in ({}, {"details": None}, {"details": []}):
            with self.subTest(metadata=metadata):
                self.requests.clear()
                self.metadata = metadata
                events = list(ResearchAssistant(client=self.client).stream("Merhaba"))
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(events[-1]["error"], "unverified_local_model")

    def test_network_timeout_has_an_explicit_error(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timeout", request=request)

        with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
            events = list(ResearchAssistant(client=client).stream("Merhaba"))
        self.assertEqual(events[-1]["error"], "ollama_unavailable")
        self.assertEqual(events[-1]["text"], "")

    def test_missing_model_does_not_pull_or_retry(self) -> None:
        self.reply = httpx.Response(404, json={"error": "model missing"})
        events = list(ResearchAssistant(client=self.client).stream("Merhaba"))
        self.assertEqual(events[-1]["error"], "model_missing")
        self.assertTrue(all(request.url.path != "/api/pull" for request in self.requests))

    def test_redirect_is_not_followed(self) -> None:
        self.reply = httpx.Response(307, headers={"Location": "https://example.org/api/chat"})
        events = list(ResearchAssistant(client=self.client).stream("Merhaba"))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(events[-1]["error"], "ollama_unavailable")

    def test_cancel_before_start_makes_no_network_calls(self) -> None:
        stop = threading.Event()
        stop.set()
        events = list(ResearchAssistant(client=self.client).stream("Merhaba", stop_event=stop))
        self.assertFalse(self.requests)
        self.assertTrue(events[-1]["cancelled"])
        self.assertIsNone(events[-1]["error"])

    def test_cancel_between_deltas_drops_remaining_text(self) -> None:
        stop = threading.Event()
        stream = ResearchAssistant(client=self.client).stream("Merhaba", stop_event=stop)
        events = []
        for event in stream:
            events.append(event)
            if event["type"] == "text_delta":
                stop.set()
        self.assertEqual(events[-1]["text"], "Merhaba")
        self.assertTrue(events[-1]["cancelled"])

    def test_cancel_after_status_prevents_search(self) -> None:
        stop = threading.Event()
        search = Mock()
        stream = ResearchAssistant(client=self.client, search_provider=search).stream(
            "Merhaba", use_web=True, stop_event=stop
        )
        for event in stream:
            if event.get("message", "").startswith("Sorunuz internette"):
                stop.set()
        search.assert_not_called()

    def test_split_urls_and_markdown_do_not_reach_speech(self) -> None:
        self.reply = _reply(
            {"message": {"content": "Bakın **bu** [kaynağa](ht"}},
            {"message": {"content": "tps://example.org). "}},
            {"message": {"content": "www.ex"}},
            {"message": {"content": "ample.com"}, "done": True},
        )
        events = list(ResearchAssistant(client=self.client).stream("Merhaba"))
        self.assertEqual(events[-1]["text"], "Bakın bu kaynağa.")
        for event in events:
            if event["type"] == "text_delta":
                self.assertNotIn("example", event["text"])
                self.assertNotIn("*", event["text"])

    def test_broken_json_or_premature_stream_end_is_not_success(self) -> None:
        for response in (
            httpx.Response(200, text="bad json\n"),
            _reply({"message": {"content": "Eksik yanıt "}}),
            _reply({"error": "runtime failed"}),
        ):
            with self.subTest(response=response):
                self.reply = response
                events = list(ResearchAssistant(client=self.client).stream("Merhaba"))
                self.assertIsNotNone(events[-1]["error"])

    def test_empty_oversized_or_invalid_requests_do_not_use_network(self) -> None:
        assistant = ResearchAssistant(client=self.client)
        for question in ("", "  ", "a" * 1001, None):
            with self.subTest(question=question):
                events = list(assistant.stream(question))
                self.assertEqual(events[-1]["error"], "invalid_question")
        events = list(assistant.stream("Merhaba", use_web="true"))
        self.assertEqual(events[-1]["error"], "invalid_question")
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
