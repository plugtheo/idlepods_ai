"""
Tests for the Gateway QueryRouter.

Covers:
- Intent classification for all 7 intent types
- Complexity detection (simple / moderate / complex)
- Agent chain lookup
- Edge cases: empty prompt, unknown fallback
"""
import pytest


@pytest.fixture
def router():
    from services.gateway.app.routing.query_router import QueryRouter
    return QueryRouter()


class TestIntentClassification:
    def test_coding_intent(self, router):
        d = router.route("implement a binary search function in Python")
        assert d.intent == "coding"

    def test_debugging_intent(self, router):
        d = router.route("there is a bug in my code causing an exception")
        assert d.intent == "debugging"

    def test_research_intent(self, router):
        d = router.route("what is the difference between TCP and UDP")
        assert d.intent == "research"

    def test_analysis_intent(self, router):
        d = router.route("analyze this module and evaluate its quality")
        assert d.intent == "analysis"

    def test_planning_intent(self, router):
        d = router.route("design the architecture for a microservices system")
        assert d.intent == "planning"

    def test_qa_intent(self, router):
        d = router.route("how does garbage collection work?")
        assert d.intent == "qa"

    def test_general_intent_fallback(self, router):
        d = router.route("x")
        assert d.intent == "general"


class TestComplexityClassification:
    def test_simple_keyword(self, router):
        d = router.route("write a quick snippet to reverse a string")
        assert d.complexity == "simple"

    def test_complex_keyword(self, router):
        d = router.route("implement a comprehensive production-ready auth system")
        assert d.complexity == "complex"

    def test_simple_by_word_count(self, router):
        d = router.route("write factorial")
        assert d.complexity == "simple"

    def test_complex_by_word_count(self, router):
        words = " ".join(["word"] * 85)
        d = router.route(f"implement a function {words}")
        assert d.complexity == "complex"

    def test_moderate_default(self, router):
        # 26-79 words, no complexity markers → moderate
        d = router.route(
            "implement a function that reads rows from a CSV file and writes filtered "
            "results to a new output file based on column values provided at runtime "
            "via command line arguments with clear error messages for invalid inputs"
        )
        assert d.complexity == "moderate"


class TestChainLookup:
    def test_coding_simple_chain(self, router):
        d = router.route("write a quick function to add two numbers")
        assert d.intent == "coding"
        assert d.complexity == "simple"
        assert d.agent_chain == ["coder", "reviewer"]

    def test_debugging_moderate_chain(self, router):
        d = router.route("fix the bug causing a NameError in the user module")
        assert d.intent == "debugging"
        assert "debugger" in d.agent_chain

    def test_research_simple_chain(self, router):
        d = router.route("what is quicksort")
        assert d.intent == "research"
        assert "researcher" in d.agent_chain

    def test_chain_is_list_of_strings(self, router):
        d = router.route("implement a REST API")
        assert isinstance(d.agent_chain, list)
        assert all(isinstance(r, str) for r in d.agent_chain)


class TestRouteDecision:
    def test_returns_route_decision(self, router):
        from services.gateway.app.routing.query_router import RouteDecision
        d = router.route("write a function")
        assert isinstance(d, RouteDecision)
        assert d.intent
        assert d.complexity
        assert isinstance(d.matched_keywords, list)
