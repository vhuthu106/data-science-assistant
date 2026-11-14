"""The ONE place that talks to the language model.

Everything else calls `chat()`, so API keys, the model name, timeouts and retries
are configured here and nowhere else.

Configuration comes from environment variables (put them in a .env file):
    OPENAI_API_KEY   required
    OPENAI_MODEL     optional, defaults to DEFAULT_MODEL
    OPENAI_BASE_URL  optional, for any OpenAI-compatible provider (read by the SDK itself)
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENAI_API_KEY is not configured."""


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # The SDK already retries rate-limit and server errors (with back-off) twice.
    # Its default read timeout is 10 minutes, far too long for an interactive app.
    return OpenAI(timeout=float(os.getenv("LLM_TIMEOUT", "60")), max_retries=2)


def chat(messages: list[dict], **kwargs) -> str:
    """Send a chat request and return the reply text."""
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=messages,
        temperature=0,   # deterministic as possible: same question -> same plan/code
        **kwargs,
    )
    return response.choices[0].message.content or ""
