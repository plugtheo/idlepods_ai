"""Round-trip tests for plans reader/writer."""
import json
import tempfile
from pathlib import Path

import pytest

from orchestration.app.plans.schema import Plan, PlanStep
from orchestration.app.plans.reader import parse_plan
from orchestration.app.plans.writer import render_plan, write_plan_atomic, validate_transition


_FIXTURE_MD = """\
## Task Goal

Fix the authentication bug in the login flow.

## Steps

1. Investigate the error logs
2. Identify the root cause
3. Write the fix
4. Add regression tests

## Constraints/Notes

Keep backward compat.
"""


def test_parse_plan_extracts_goal():
    plan = parse_plan(_FIXTURE_MD)
    assert "Fix the authentication bug" in plan.goal


def test_parse_plan_extracts_steps():
    plan = parse_plan(_FIXTURE_MD)
    assert len(plan.steps) == 4
    assert plan.steps[0].id == "step-1"
    assert "error logs" in plan.steps[0].description
    assert plan.steps[3].id == "step-4"


def test_parse_plan_all_steps_pending():
    plan = parse_plan(_FIXTURE_MD)
    assert all(s.status == "pending" for s in plan.steps)


def test_render_plan_round_trip():
    plan = parse_plan(_FIXTURE_MD)
    rendered = render_plan(plan)
    # Idempotent: re-parsing the rendered output must yield the same steps
    plan2 = parse_plan(rendered)
    assert [s.description for s in plan.steps] == [s.description for s in plan2.steps]
    assert plan.goal == plan2.goal


def test_write_plan_atomic(tmp_path):
    plan = parse_plan(_FIXTURE_MD)
    out = tmp_path / "plans" / "current-task.md"
    out.parent.mkdir(parents=True)
    write_plan_atomic(out, plan)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Fix the authentication bug" in content
    assert "step-1" not in content  # ids are internal, not written literally
    assert "Investigate the error logs" in content


def test_no_tmp_file_left_after_write(tmp_path):
    plan = parse_plan(_FIXTURE_MD)
    out = tmp_path / "plans" / "current-task.md"
    out.parent.mkdir(parents=True)
    write_plan_atomic(out, plan)
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


def test_validate_transition_allows_pending_to_in_progress():
    plan = parse_plan(_FIXTURE_MD)
    import copy
    new_plan = copy.deepcopy(plan)
    new_plan.steps[0].status = "in_progress"
    validate_transition(plan, new_plan)  # must not raise


def test_validate_transition_rejects_done_to_pending():
    plan = parse_plan(_FIXTURE_MD)
    import copy
    done_plan = copy.deepcopy(plan)
    done_plan.steps[0].status = "done"
    back_plan = copy.deepcopy(done_plan)
    back_plan.steps[0].status = "pending"
    with pytest.raises(ValueError, match="Illegal status transition"):
        validate_transition(done_plan, back_plan)


def test_duplicate_step_ids_rejected():
    with pytest.raises(Exception):
        Plan(
            goal="test",
            steps=[
                PlanStep(id="step-1", description="a"),
                PlanStep(id="step-1", description="b"),
            ],
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
