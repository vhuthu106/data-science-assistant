"""app.py: the Streamlit app, driven like a user would (Steps 3-8). Uses Streamlit's AppTest."""
import json

import pytest
from streamlit.testing.v1 import AppTest

import agents.planner as planner_module
import agents.worker as worker_module
from conftest import REPO, fake_llm

PLAN = json.dumps({"intent": "avg_age", "steps": [{"action": "group_analysis", "args": {"query": "Mean Age per Pclass"}}]})


@pytest.fixture
def app():
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=60).run()
    at.sidebar.button[0].click().run()          # "Load Titanic Dataset"
    return at


def ask(at, question):
    at.text_input[0].input(question).run()
    next(b for b in at.button if b.label == "Run Analysis").click().run()


def test_dataset_survives_reruns(app):
    # Regression: the dataset used to vanish on the next interaction.
    app.run()
    app.run()
    assert "📊 Dataset Preview" in [s.value for s in app.subheader]


def test_question_end_to_end(app, monkeypatch):
    monkeypatch.setattr(planner_module, "chat", fake_llm(PLAN))
    monkeypatch.setattr(worker_module, "chat", fake_llm("result = df.groupby('Pclass')['Age'].mean().round(1)"))
    ask(app, "average age by passenger class")
    assert not app.exception and not app.error
    tables = [t.value.squeeze().to_dict() for t in app.dataframe if t.value.shape[0] == 3]
    assert {1: 38.2, 2: 29.9, 3: 25.1} in tables


def test_failed_run_is_logged_as_failed(app, monkeypatch):
    monkeypatch.setattr(planner_module, "chat", fake_llm(PLAN))
    monkeypatch.setattr(worker_module, "chat", fake_llm("result = df['NoSuchColumn'].sum()"))
    ask(app, "this will fail")
    from core.run_log import RunLog
    last = RunLog().load(limit=1).iloc[0]
    assert (last["question"], last["status"]) == ("this will fail", "failed")
