"""core/llm.py: provider errors become clear messages (Step 6).

These go through the REAL OpenAI SDK, talking to a tiny fake server on this machine.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import core.llm as llm


class FakeProvider(BaseHTTPRequestHandler):
    status, body, delay = 200, {}, 0

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(FakeProvider.delay)
        data = json.dumps(FakeProvider.body).encode()
        try:
            self.send_response(FakeProvider.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass   # the client already gave up (timeout test)

    def log_message(self, *args):
        pass


@pytest.fixture
def provider(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT", "1")
    llm.get_client.cache_clear()
    yield FakeProvider
    server.shutdown()
    llm.get_client.cache_clear()


def error(message):
    return {"error": {"message": message}}


@pytest.mark.parametrize("status, body, delay, expected", [
    (429, error("Rate limit reached"), 0, "rate limit"),
    (429, error("You exceeded your current quota"), 0, "quota or credit"),
    (503, error("high demand"), 0, "overloaded"),
    (401, error("Invalid API key"), 0, "API key was rejected"),
    (404, error("model not found"), 0, "'test-model' was not found"),
    (200, {}, 2, "did not answer in time"),
], ids=["rate limit", "quota", "overloaded", "bad key", "unknown model", "timeout"])
def test_provider_errors_become_clear_messages(provider, status, body, delay, expected):
    provider.status, provider.body, provider.delay = status, body, delay
    with pytest.raises(llm.LLMUnavailable, match=expected):
        llm.chat([{"role": "user", "content": "hi"}])


def test_unreachable_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")   # nothing listens here
    llm.get_client.cache_clear()
    with pytest.raises(llm.LLMUnavailable, match="Could not reach"):
        llm.chat([{"role": "user", "content": "hi"}])
    llm.get_client.cache_clear()
