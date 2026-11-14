"""agents/worker.py: code generation, self-repair and stopping cleanly (Steps 5-7)."""
import agents.worker as worker_module
from agents.worker import Worker, extract_code
from core.llm import LLMUnavailable
from conftest import fake_llm


def make_plan(*instructions):
    return {"question": "q", "steps": [{"action": "data_analysis", "args": {"query": i}} for i in instructions]}


def test_failed_code_is_repaired_using_the_error(monkeypatch, dataset_manager):
    fake = fake_llm("result = df.groupby('Pclass')['Agee'].mean()",
                    "result = df.groupby('Pclass')['Age'].mean().round(1)")
    monkeypatch.setattr(worker_module, "chat", fake)
    step = Worker().handle(make_plan("mean age per class"), dataset_manager)["steps"][0]

    assert step["success"] and list(step["result"]) == [38.2, 29.9, 25.1]
    assert len(step["failed_attempts"]) == 1
    repair_request = fake.calls[1]
    assert "'Agee'" in repair_request[-2]["content"]           # the model sees its own code...
    assert "KeyError (line 1)" in repair_request[-1]["content"]  # ...and the exact error
    assert "mean age per class" in repair_request[1]["content"]  # ...and the original task


def test_gives_up_after_repairs_and_skips_later_steps(monkeypatch, dataset_manager):
    monkeypatch.setattr(worker_module, "chat", fake_llm("result = df['NoSuchColumn'].sum()"))
    out = Worker().handle(make_plan("broken", "plot something"), dataset_manager)
    assert not out["steps"][0]["success"] and len(out["steps"][0]["failed_attempts"]) == 3
    assert out["steps"][1]["skipped"]
    assert "Step 1 failed" in out["stopped"]


def test_provider_error_keeps_finished_steps(monkeypatch, dataset_manager):
    fake = fake_llm("result = len(df)", LLMUnavailable("rate limit"), "result = 1")
    monkeypatch.setattr(worker_module, "chat", fake)
    out = Worker().handle(make_plan("count rows", "second", "third"), dataset_manager)
    assert out["steps"][0]["success"] and out["steps"][0]["result"] == 891
    assert out["steps"][1]["error"] == "rate limit"
    assert out["steps"][2]["skipped"]


def test_blocked_code_goes_through_repair(monkeypatch, dataset_manager):
    fake = fake_llm("import os\nresult = df.to_csv('out.csv')",
                    "result = df.groupby('Sex')['Survived'].mean().round(2)")
    monkeypatch.setattr(worker_module, "chat", fake)
    step = Worker().handle(make_plan("survival rate by sex"), dataset_manager)["steps"][0]
    assert step["success"] and step["result"].to_dict() == {"female": 0.74, "male": 0.19}
    assert step["failed_attempts"][0]["error"].startswith("Blocked for safety")


def test_extract_code():
    assert extract_code("Here you go:\n```python\nresult = 1\n```\nDone!") == "result = 1"
    assert extract_code("result = 1") == "result = 1"
