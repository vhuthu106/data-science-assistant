"""agents/planner.py: turning a question into a validated plan (Step 4)."""
import json

import pytest

import agents.planner as planner_module
from agents.planner import Planner
from conftest import fake_llm

GOOD = {"intent": "avg_age", "steps": [{"action": "group_analysis", "args": {"query": "Mean Age per Pclass"}}]}
QUESTION = "average age by passenger class"


def plan_with(monkeypatch, titanic, reply):
    fake = fake_llm(reply)
    monkeypatch.setattr(planner_module, "chat", fake)
    return Planner().plan(QUESTION, titanic), fake


@pytest.mark.parametrize("reply", [
    json.dumps(GOOD),
    "Sure! Here is the plan:\n```json\n" + json.dumps(GOOD) + "\n```\nHope it helps.",
], ids=["clean JSON", "JSON in fences with chatter"])
def test_valid_plans_are_used(monkeypatch, titanic, reply):
    plan, _ = plan_with(monkeypatch, titanic, reply)
    assert plan["steps"] == GOOD["steps"] and "fallback_reason" not in plan
    assert plan["question"] == QUESTION


@pytest.mark.parametrize("reply, reason", [
    ("I think you should look at the ages.", "no JSON object"),
    (json.dumps({"steps": [{"action": "train_neural_net", "args": {"query": "x"}}]}), "unknown action"),
    (json.dumps({"steps": [{"action": "create_plot", "args": {}}]}), "no instruction"),
    (json.dumps({"intent": "x"}), "no steps"),
])
def test_unusable_plans_fall_back(monkeypatch, titanic, reply, reason):
    plan, _ = plan_with(monkeypatch, titanic, reply)
    assert reason in plan["fallback_reason"]
    assert plan["steps"][0]["action"] == "group_analysis"   # the rule-based guess for this question


def test_long_plans_are_cut_to_three_steps(monkeypatch, titanic):
    reply = json.dumps({"steps": [{"action": "data_analysis", "args": {"query": f"s{i}"}} for i in range(5)]})
    plan, _ = plan_with(monkeypatch, titanic, reply)
    assert len(plan["steps"]) == 3


def test_prompt_has_columns_but_no_row_values(monkeypatch, titanic):
    _, fake = plan_with(monkeypatch, titanic, json.dumps(GOOD))
    prompt = fake.calls[0][-1]["content"]
    assert "Pclass" in prompt and "Age: float64" in prompt and "891 rows" in prompt
    assert "Braund" not in prompt   # a passenger's name: row data must never be sent


def test_provider_errors_are_not_hidden(monkeypatch, titanic):
    with pytest.raises(TimeoutError):
        plan_with(monkeypatch, titanic, TimeoutError("Request timed out."))


@pytest.mark.parametrize("question, action", [
    ("average age by passenger class", "group_analysis"),
    ("plot the ages", "create_plot"),
    ("show missing values", "data_cleaning"),
    ("describe the data", "data_analysis"),
])
def test_rule_based_plan(question, action):
    assert Planner.rule_based_plan(question)["steps"][0]["action"] == action
