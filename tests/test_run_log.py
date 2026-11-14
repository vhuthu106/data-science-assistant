"""core/run_log.py: honest metrics (Step 8)."""
from core.run_log import RunLog, run_status


def step(success, failed_attempts=0, skipped=False):
    return {"success": success, "skipped": skipped, "failed_attempts": [{}] * failed_attempts}


def test_all_steps_ok_is_success():
    assert run_status({"steps": [step(True), step(True)]})["status"] == "success"


def test_failed_run_is_not_counted_as_success():
    # Regression: the old app logged this as "Analysis completed successfully".
    s = run_status({"steps": [step(False, 3), step(False, skipped=True)], "stopped": "Step 1 failed"})
    assert s["status"] == "failed" and s["failed_attempts"] == 3 and s["error"] == "Step 1 failed"


def test_some_steps_ok_is_partial():
    assert run_status({"steps": [step(True), step(False, 3)]})["status"] == "partial"


def test_repair_counts():
    s = run_status({"steps": [step(True, 1), step(True), step(False, 3)]})
    assert (s["repaired_steps"], s["repaired_ok"]) == (2, 1)


def test_record_load_and_summary(tmp_path):
    log = RunLog(tmp_path / "runs.db")
    log.record("q1", "m", "success", 1.0, steps_total=1, steps_ok=1)
    log.record("q2", "m", "failed", 2.0, steps_total=1, failed_attempts=3, repaired_steps=1)
    log.record("q3", "m", "success", 3.0, steps_total=1, steps_ok=1, failed_attempts=1, repaired_steps=1, repaired_ok=1)
    log.record("q4", "m", "provider_error", 0.5, error="rate limit")
    s = RunLog.summary(log.load())
    assert s["runs"] == 4 and s["success_rate"] == 0.5
    assert s["provider_errors"] == 1 and s["repair_success_rate"] == 0.5
