"""
Shared quality-signal regexes used for adapter output validation.

Import from here instead of redefining per-file.  Any update to the pattern
automatically propagates to orchestration scoring, training data generation,
bootstrap validation, eval scripts, and e2e tests.
"""

import re

# Matches pipeline metadata keys that should never appear in raw model output.
# Adapter contamination occurs when training data contains ExperienceEvent JSON
# and the adapter "memorises" the schema rather than learning the task domain.
METADATA_LEAKAGE_RE = re.compile(
    r"'agent_name'\s*:|\"agent_name\"\s*:"
    r"|'quality_score'\s*:|\"quality_score\"\s*:"
    r"|'iteration_number'\s*:|\"iteration_number\"\s*:"
    r"|'execution_time_ms'\s*:|\"execution_time_ms\"\s*:"
    r"|'session_id'\s*:|\"session_id\"\s*:"
    r"|'final_output'\s*:|\"final_output\"\s*:"
    r"|'agent_chain'\s*:|\"agent_chain\"\s*:"
    r"|'contributions'\s*:|\"contributions\"\s*:"
    r"|'iteration_scores'\s*:|\"iteration_scores\"\s*:"
    r"|'final_score'\s*:|\"final_score\"\s*:"
    r"|'converged'\s*:|\"converged\"\s*:"
    r"|convergence_threshold",
    re.IGNORECASE,
)

# Matches GPT-2 byte-level BPE unicode placeholders left in adapter output.
#   Ġ (U+0120) = space  |  Ċ (U+010A) = newline  |  ▁ (U+2581) = sentencepiece space
BPE_ARTIFACT_RE = re.compile(r"[ĠĊ▁]")
