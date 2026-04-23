"""
Tests for the Training Service experience reader.

Covers all three diversity criteria (pass and fail for each).
"""
import json
import pytest
from unittest.mock import patch


def _make_records(n=60, spread=0.30, diversity=1.0):
    """
    Build a list of synthetic experience records.

    n        — number of records
    spread   — difference between max and min score
    diversity— ratio of unique prompts (1.0 = all unique)
    """
    records = []
    unique_count = max(1, int(n * diversity))
    for i in range(n):
        prompt_idx = i % unique_count
        score = 0.5 + (i / n) * spread  # scores from 0.5 to 0.5+spread
        records.append({
            "prompt": f"unique prompt number {prompt_idx} doing something",
            "final_output": f"output {i}",
            "final_score": score,
        })
    return records


class TestCheckDiversity:
    def _run(self, records, min_batch=50, spread=0.15, ratio=0.60):
        mock_settings = type("S", (), {
            "min_batch_size": min_batch,
            "min_score_spread": spread,
            "min_diversity_ratio": ratio,
            "min_quality_score": 0.0,
            "jsonl_path": "/tmp/fake.jsonl",
        })()
        with patch("services.training.app.utils.experience_reader.settings", mock_settings):
            from services.training.app.utils.experience_reader import check_diversity
            return check_diversity(records)

    def test_passes_when_all_criteria_met(self):
        records = _make_records(n=60, spread=0.30, diversity=1.0)
        passed, reason = self._run(records)
        assert passed, f"Expected pass but got: {reason}"

    def test_fails_when_too_few_records(self):
        records = _make_records(n=10)
        passed, reason = self._run(records, min_batch=50)
        assert not passed
        assert "too few" in reason

    def test_fails_when_score_spread_too_small(self):
        # All scores nearly identical → tiny spread
        records = [{"prompt": f"p{i}", "final_score": 0.75} for i in range(60)]
        passed, reason = self._run(records, spread=0.15)
        assert not passed
        assert "spread" in reason

    def test_fails_when_low_diversity(self):
        # Only 10% unique prompts
        records = _make_records(n=60, spread=0.30, diversity=0.1)
        passed, reason = self._run(records, ratio=0.60)
        assert not passed
        assert "diversity" in reason

    def test_reason_string_contains_stats_on_pass(self):
        records = _make_records(n=60, spread=0.30, diversity=1.0)
        passed, reason = self._run(records)
        assert passed
        assert "n=" in reason
        assert "spread=" in reason


class TestLoadExperiences:
    def test_returns_empty_list_if_no_file(self, tmp_path):
        mock_settings = type("S", (), {"jsonl_path": str(tmp_path / "no.jsonl")})()
        with patch("services.training.app.utils.experience_reader.settings", mock_settings):
            from services.training.app.utils.experience_reader import load_experiences
            records = load_experiences()
        assert records == []

    def test_reads_all_lines(self, tmp_path):
        jsonl = tmp_path / "exp.jsonl"
        for i in range(5):
            jsonl.write_text(
                "\n".join(
                    json.dumps({"prompt": f"p{j}", "final_score": 0.8}) for j in range(5)
                )
            )

        mock_settings = type("S", (), {"jsonl_path": str(jsonl)})()
        with patch("services.training.app.utils.experience_reader.settings", mock_settings):
            from services.training.app.utils.experience_reader import load_experiences
            records = load_experiences()
        assert len(records) == 5

    def test_skips_invalid_json_lines(self, tmp_path):
        jsonl = tmp_path / "exp.jsonl"
        jsonl.write_text(
            '{"prompt": "ok", "final_score": 0.8}\n'
            'not-json\n'
            '{"prompt": "ok2", "final_score": 0.9}\n'
        )

        mock_settings = type("S", (), {"jsonl_path": str(jsonl)})()
        with patch("services.training.app.utils.experience_reader.settings", mock_settings):
            from services.training.app.utils.experience_reader import load_experiences
            records = load_experiences()
        assert len(records) == 2


class TestFingerprint:
    def test_normalises_case_and_whitespace(self):
        from services.training.app.utils.experience_reader import _fingerprint
        a = _fingerprint("Hello   World")
        b = _fingerprint("hello world")
        assert a == b

    def test_truncated_to_120_chars(self):
        from services.training.app.utils.experience_reader import _fingerprint
        long_text = "a" * 200
        assert len(_fingerprint(long_text)) <= 120
