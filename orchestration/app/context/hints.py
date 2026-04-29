"""
System hint generation
========================
Generates brief guidance text injected into every agent's system prompt for
the duration of a request, based on the classified intent and complexity.

Active for CODING, DEBUGGING, RESEARCH, and ANALYSIS intents.
All other intents return an empty string.
"""

from __future__ import annotations


async def generate(intent: str, complexity: str) -> str:
    """Return a brief guidance string for the given intent and complexity."""
    return _build_hints(intent, complexity)


def _build_hints(intent: str, complexity: str) -> str:
    hints: list[str] = []

    if intent == "coding":
        hints.append("Write well-structured, production-quality code with clear naming.")
    elif intent == "debugging":
        hints.append("Identify the root cause before proposing a fix. Show the fix clearly.")
    elif intent == "research":
        hints.append("Cite sources or reasoning. Be accurate and concise.")
    elif intent == "planning":
        hints.append("Break the task into concrete, actionable steps.")
    elif intent == "analysis":
        hints.append("Be systematic. Support conclusions with evidence from the data or code.")

    if complexity == "complex":
        hints.append("Consider edge cases, scalability, and maintainability.")
    elif complexity == "simple":
        hints.append("Keep the response focused and concise.")

    return "  ".join(hints)
