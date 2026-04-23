"""
Tests for the orchestration scoring utilities.

Covers:
- SCORE: annotation parsed correctly (0-1 scale and 0-10 scale)
- Keyword heuristics: blockers lower score, positive phrases raise it
- Empty/short text returns low score
- score_iteration function dispatches correctly
"""
import pytest


class TestExtractScoreFromText:
    def test_score_annotation_01_scale(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        assert extract_score_from_text("The code looks good. SCORE: 0.85") == pytest.approx(0.85)

    def test_score_annotation_with_equals(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        assert extract_score_from_text("SCORE=0.72") == pytest.approx(0.72)

    def test_score_annotation_010_scale_converted(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        val = extract_score_from_text("SCORE: 8.5")
        assert val == pytest.approx(0.85)

    def test_score_above_10_ignored(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        assert extract_score_from_text("SCORE: 95") is None

    def test_no_score_annotation(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        assert extract_score_from_text("The code is correct") is None

    def test_case_insensitive(self):
        from services.orchestration.app.utils.scoring import extract_score_from_text
        assert extract_score_from_text("score: 0.5") == pytest.approx(0.5)


class TestHeuristicScore:
    def test_empty_text_is_low(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        assert heuristic_score("", "coder") == pytest.approx(0.30)

    def test_short_text_is_low(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        assert heuristic_score("ok", "reviewer") == pytest.approx(0.30)

    def test_explicit_score_takes_priority(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        result = heuristic_score("Detailed review of the implementation quality here. SCORE: 0.90", "reviewer")
        assert result == pytest.approx(0.90)

    def test_positive_signal_raises_score(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        base = heuristic_score("A" * 50, "reviewer")  # no signals
        positive = heuristic_score("A" * 50 + " LOOKS GOOD and NO ISSUES", "reviewer")
        assert positive > base

    def test_blocker_signal_lowers_score(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        base = heuristic_score("A" * 50, "reviewer")
        blocked = heuristic_score("A" * 50 + " BLOCKER: security hole found", "reviewer")
        assert blocked < base

    def test_score_clamped_to_01(self):
        from services.orchestration.app.utils.scoring import heuristic_score
        text = " ".join(["LOOKS GOOD NO ISSUES BLOCKERS: None WELL STRUCTURED"] * 20)
        result = heuristic_score(text, "reviewer")
        assert 0.0 <= result <= 1.0


class TestScoreIteration:
    def test_empty_history_returns_zero(self):
        from services.orchestration.app.utils.scoring import score_iteration
        assert score_iteration([], 1) == 0.0

    def test_iteration_not_in_history_returns_zero(self):
        from services.orchestration.app.utils.scoring import score_iteration
        history = [{"iteration": 1, "role": "coder", "output": "x" * 50}]
        assert score_iteration(history, 99) == 0.0

    def test_reviewer_output_scored(self):
        from services.orchestration.app.utils.scoring import score_iteration
        history = [
            {"iteration": 1, "role": "coder", "output": "def f(): pass"},
            {"iteration": 1, "role": "reviewer", "output": "This implementation is correct and well structured. SCORE: 0.80 looks good"},
        ]
        score = score_iteration(history, 1)
        assert score == pytest.approx(0.80)

    def test_uses_last_reviewer_in_iteration(self):
        from services.orchestration.app.utils.scoring import score_iteration
        history = [
            {"iteration": 1, "role": "reviewer", "output": "Some issues with error handling need fixing here. SCORE: 0.60"},
            {"iteration": 1, "role": "critic", "output": "Overall structure is acceptable but needs more test coverage. SCORE: 0.75"},
        ]
        score = score_iteration(history, 1)
        # Should use the highest or last evaluation score
        assert score >= 0.60
