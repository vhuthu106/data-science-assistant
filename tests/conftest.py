"""Shared test setup. No test ever calls a real model: the LLM is always replaced by a fake."""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Set BEFORE the app's modules are imported:
os.environ.setdefault("OPENAI_API_KEY", "test-key")            # CI has no real key, and needs none
os.environ["RUNS_DB"] = str(Path(tempfile.mkdtemp()) / "runs.db")  # never touch your real runs.db

REPO = Path(__file__).resolve().parent.parent
TITANIC = REPO / "datasets" / "Titanic-Dataset.csv"


@pytest.fixture
def titanic():
    return pd.read_csv(TITANIC)


@pytest.fixture
def dataset_manager():
    from tools.dataset_tools import DatasetManager
    dm = DatasetManager()
    dm.load_from_path("titanic", TITANIC)
    return dm


def fake_llm(*replies):
    """A stand-in for core.llm.chat that returns the given replies in order and records every call."""
    calls = []
    replies = list(replies)

    def chat(messages, **kwargs):
        calls.append(messages)
        reply = replies.pop(0) if len(replies) > 1 else replies[0]   # last reply repeats
        if isinstance(reply, Exception):
            raise reply
        return reply

    chat.calls = calls
    return chat
