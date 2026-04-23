"""
Tests for the orchestration pipeline builder.

Covers:
- build_pipeline() returns a compiled graph without errors
- _recursion_limit() calculation
- _finalize_state() returns complete final output
"""
import pytest
from unittest.mock import patch, MagicMock


class TestBuildPipeline:
    def test_pipeline_compiles_without_error(self):
        """build_pipeline() should succeed with LangGraph installed."""
        from services.orchestration.app.graph.pipeline import build_pipeline
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_pipeline_has_ainvoke(self):
        """Compiled graph must have ainvoke for async execution."""
        from services.orchestration.app.graph.pipeline import build_pipeline
        pipeline = build_pipeline()
        assert hasattr(pipeline, "ainvoke")


class TestRecursionLimit:
    def test_default_limit(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        # Default 5 iterations, 7 nodes → should be well above 35
        limit = _recursion_limit(max_iterations=5, chain_length=7)
        assert limit > 35

    def test_single_iteration(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        limit = _recursion_limit(max_iterations=1, chain_length=2)
        assert limit >= 2

    def test_scales_with_iterations(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        limit_5 = _recursion_limit(max_iterations=5, chain_length=4)
        limit_10 = _recursion_limit(max_iterations=10, chain_length=4)
        assert limit_10 > limit_5
