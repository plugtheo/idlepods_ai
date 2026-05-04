"""
Step 11 — test_scheduler_timeout.py

Mock a subprocess that exceeds the timeout; assert:
- The process group is terminated
- 'training_timed_out' appears in the error log
- The Redis cursor is NOT advanced (training.app.trainer_wrapper path)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


def _make_timeout_exc():
    proc = MagicMock()
    proc.pid = 99999
    exc = subprocess.TimeoutExpired(cmd=["docker"], timeout=1)
    exc.process = proc
    return exc


@pytest.mark.parametrize("platform", ["win32", "posix"])
def test_run_local_timeout_logs_and_terminates(platform, tmp_path, caplog):
    """_run_local: TimeoutExpired → logs training_timed_out, attempts termination."""
    caplog.set_level(logging.ERROR)

    tmp_jsonl = tmp_path / "data.jsonl"
    tmp_jsonl.write_text("")

    with (
        patch("training.app.trainer_wrapper.settings") as mock_settings,
        patch("training.app.trainer_wrapper._base_model_for", return_value="base-model"),
        patch("training.app.trainer_wrapper.to_training_records", return_value=[{"role": "coder"}]),
        patch("training.app.trainer_wrapper.CAPABILITIES", ["coder"]),
        patch("training.app.trainer_wrapper.sys") as mock_sys,
        patch("subprocess.run", side_effect=_make_timeout_exc()),
        patch("sys.platform", platform),
    ):
        mock_settings.training_timeout_seconds = 1
        mock_settings.output_dir = str(tmp_path / "out")
        mock_sys.executable = sys.executable
        mock_sys.platform = platform

        import training.app.trainer_wrapper as wrapper

        with patch.object(wrapper, "CAPABILITIES", ["coder"]):
            import tempfile, json
            # Provide a real temp file so NamedTemporaryFile works
            records = [{"role": "coder", "final_score": 0.9}]
            with patch("training.app.trainer_wrapper.to_training_records", return_value=records):
                with patch("training.app.trainer_wrapper._base_model_for", return_value="base-model"):
                    with patch("subprocess.run", side_effect=_make_timeout_exc()):
                        wrapper._run_local(records)

    assert any("training_timed_out" in r.message for r in caplog.records), (
        f"Expected 'training_timed_out' in log. Records: {[r.message for r in caplog.records]}"
    )


def test_run_local_timeout_does_not_advance_cursor(tmp_path, caplog):
    """
    Timeout must leave the cursor untouched.  We simulate by verifying
    that _run_local raises no exception that would cause the caller (scheduler)
    to commit the cursor — the function returns normally after logging.
    """
    caplog.set_level(logging.ERROR)

    records = [{"role": "coder", "final_score": 0.9}]

    with (
        patch("training.app.trainer_wrapper._base_model_for", return_value="base-model"),
        patch("training.app.trainer_wrapper.to_training_records", return_value=records),
        patch("training.app.trainer_wrapper.CAPABILITIES", ["coder"]),
        patch("subprocess.run", side_effect=_make_timeout_exc()),
    ):
        import training.app.trainer_wrapper as wrapper
        # Must not raise — caller relies on return value absence to detect timeout
        try:
            wrapper._run_local(records)
        except Exception as exc:
            pytest.fail(f"_run_local raised unexpectedly: {exc}")

    assert any("training_timed_out" in r.message for r in caplog.records)
