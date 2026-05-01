"""
InferenceOptimizer
==================
Switchable token optimization levers for the multi-agent inference pipeline.

Each lever can be enabled or disabled independently via environment variables
without any code changes:

    ORCHESTRATION__OPTIMIZE_ROLE_HISTORY_FILTER=false
    ORCHESTRATION__OPTIMIZE_STRUCTURED_EXTRACTION=false

Levers
------
role_history_filter
    Each agent receives only the history entries from roles it semantically
    depends on, rather than the full accumulated history.  Reduces input token
    count for every agent except consensus.

    Example: reviewer only needs coder + debugger outputs — not planner's plan
    or researcher's prose, which are irrelevant to a code review.

structured_extraction
    Agents whose outputs follow a strict key-value format (reviewer, critic,
    debugger) have only the meaningful fields stored in iteration_history.
    The full prose response is still passed as last_output for the convergence
    scorer (which must see SCORE: annotations).  Downstream agents that read
    history get compact, token-efficient entries instead of full prose.

These two levers are orthogonal and can be combined freely.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from ..config.settings import settings

logger = logging.getLogger(__name__)


# ── Role history filter map ───────────────────────────────────────────────
#
# Maps each agent role to the set of prior roles whose outputs it actually
# needs in context.  None means the agent receives the full unfiltered
# history (no restriction).
#
# Design rationale per role:
#   planner    — usually first in chain; in subsequent iterations it needs
#                full context to understand what went wrong.
#   researcher — scoped by planner's breakdown; no need to see evaluations.
#   coder      — implements the plan using research; doesn't need reviews.
#   debugger   — fixes the code; needs the code and original intent only.
#   reviewer   — evaluates the implementation; ignores planning prose.
#   critic     — delivers a verdict informed by reviewer's structured analysis.
#   consensus  — synthesises everything; receives full history.

_ROLE_HISTORY_FILTER: Dict[str, Optional[Set[str]]] = {
    "planner":    None,
    "researcher": {"planner"},
    "coder":      {"planner", "researcher", "tool"},
    "debugger":   {"coder", "planner"},
    "reviewer":   {"coder", "debugger"},
    "critic":     {"reviewer", "planner", "researcher"},  # see all roles so planner/research chains get proper critique context
    "consensus":  None,
}


# ── Structured field definitions ─────────────────────────────────────────
#
# For roles that emit structured plain-text output, list the field labels to
# extract.  Only these fields are stored in iteration_history; the rest of
# the prose is discarded from the history entry (but kept in last_output).
#
# Field order matters — extraction follows list order, and the stored entry
# preserves that order.

_STRUCTURED_FIELDS: Dict[str, List[str]] = {
    "reviewer": ["SCORE", "ISSUES", "SUGGESTIONS"],
    "critic":   ["SCORE", "BLOCKERS", "IMPROVEMENT"],
    "debugger": ["ISSUE", "FIX"],
}

# Max characters kept per extracted field value in history entries.
# Configurable via ORCHESTRATION__STRUCTURED_FIELD_VALUE_MAX_CHARS.
# Resolved at call time from settings so a single env-var change takes effect
# after service restart without code modifications.


class InferenceOptimizer:
    """
    Token optimization levers for agent inference calls.

    Instantiated once at module level in nodes.py and shared across all
    node invocations within a process lifetime.

    Parameters
    ----------
    role_history_filter : bool
        When True, each agent only receives the history entries from roles it
        semantically depends on (see _ROLE_HISTORY_FILTER).
    structured_extraction : bool
        When True, structured-output roles (reviewer, critic, debugger) have
        only their key fields stored in iteration_history, reducing the token
        cost of history entries for downstream agents.
    """

    def __init__(
        self,
        role_history_filter: bool = True,
        structured_extraction: bool = True,
    ) -> None:
        self.role_history_filter = role_history_filter
        self.structured_extraction = structured_extraction
        logger.info(
            "InferenceOptimizer initialised: role_history_filter=%s  structured_extraction=%s",
            role_history_filter,
            structured_extraction,
        )

    # ── Lever 1: Role-filtered history ────────────────────────────────────

    def filter_history(
        self,
        role: str,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return only the history entries this agent role needs.

        When the lever is disabled the full history list is returned unchanged,
        making the pipeline behave identically to the baseline.

        Parameters
        ----------
        role : str
            The agent role about to receive this history.
        history : list
            Full accumulated iteration_history from AgentState.

        Returns
        -------
        list
            Filtered (or unmodified) history list.
        """
        if not self.role_history_filter:
            return history

        allowed = _ROLE_HISTORY_FILTER.get(role)
        if allowed is None:
            return history  # role is configured to receive everything

        filtered = [h for h in history if h.get("role") in allowed]

        if len(filtered) < len(history):
            logger.debug(
                "role=%s  history_in=%d  history_out=%d  allowed_roles=%s",
                role,
                len(history),
                len(filtered),
                sorted(allowed),
            )

        return filtered

    # ── Lever 2: Structured output extraction ─────────────────────────────

    def extract_for_history(self, role: str, full_output: str) -> str:
        """
        Extract only the key structured fields from a role's output for
        storage in iteration_history.

        The caller is responsible for keeping the full_output as last_output
        so the convergence scorer can still find SCORE: annotations.

        Falls back to the full output unchanged if:
        - the lever is disabled
        - the role has no structured field definition
        - no fields could be parsed from the output (model deviated from format)

        Parameters
        ----------
        role : str
            The agent role that produced the output.
        full_output : str
            The complete text returned by the model.

        Returns
        -------
        str
            Compact extracted string, or full_output as fallback.
        """
        if not self.structured_extraction:
            return full_output

        fields = _STRUCTURED_FIELDS.get(role)
        if not fields:
            return full_output  # role has no extractable structured format

        extracted: Dict[str, str] = {}
        for field in fields:
            # Match "FIELD: <value>" up to the next ALL-CAPS label or end of string.
            # re.DOTALL so value can span multiple lines (e.g. FIX: contains code).
            pattern = rf"(?m)^{re.escape(field)}:\s*(.+?)(?=\n[A-Z]{{2,}}:|$)"
            match = re.search(pattern, full_output, re.DOTALL)
            if match:
                value = match.group(1).strip()[:settings.structured_field_value_max_chars]
                # Collapse internal whitespace sequences to single spaces so the
                # stored string is compact but still human-readable.
                extracted[field] = re.sub(r"\s+", " ", value)

        if not extracted:
            logger.debug(
                "role=%s structured extraction found no fields — storing full output",
                role,
            )
            return full_output

        logger.debug(
            "role=%s extracted fields=%s  chars_before=%d  chars_after=%d",
            role,
            list(extracted.keys()),
            len(full_output),
            sum(len(v) for v in extracted.values()),
        )

        return "\n".join(f"{k}: {v}" for k, v in extracted.items())
