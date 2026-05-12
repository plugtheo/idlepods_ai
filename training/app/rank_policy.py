"""
Auto rank-promotion & freeze policy
====================================
Decides whether to bump an adapter's LoRA rank between retrain rounds, and —
once the rank has hit the configured cap — whether to *freeze* the adapter so
it stops accumulating training rounds and waits to be merged into a standalone
per-agent model.

Bumps and freezes both fire only when ALL three plateau signals hold across a
smoothing window of prior adapter_diff_*.json reports:

  1. Loss reduction per round  <  2 %    — the adapter has stopped improving
     measurably on its current capacity.
  2. Regression delta  <  0.005          — new versions barely beat the
     previous active adapter; head-room exhausted at this rank.
  3. Dataset growth  ≥  1.5×             — more data has accumulated since
     the last bump than the current rank can absorb at convergence.

Cooldown of `rank_promotion_cooldown` successful promotions prevents two
back-to-back bumps from a single noisy round driving a permanent decision.

The policy writes ``<adapter_dir>/runtime_recipe_override.json`` (persistent
between runs). trainer_entry.py reads the override after ``lookup_recipe()``
and merges its `r`/`alpha` into the recipe before training. The same file
records the `frozen` flag — once set, the wrapper skips this role's retrains.
recipes.yaml stays the source of truth for *intent*; the override file
records the *current applied state* of automated decisions.

Operational notes:
- To unfreeze (e.g., after merging the frozen adapter into a new per-agent
  base), delete the override file or set its `"frozen": false` key.
- The override's ``r`` / ``alpha`` describe the *current trained rank* once a
  bump has fired. ``maybe_promote_rank`` derives its "current rank" from the
  most recent history entry, so callers passing a stale YAML default to
  ``current_r`` do not cause double-bumps.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Thresholds (intentionally non-tunable from settings — these are policy
# parameters, not deployment knobs; changing them is an architectural call).
_LOSS_REDUCTION_THRESHOLD       = 0.02   # per-round, fraction
_REGRESSION_DELTA_THRESHOLD     = 0.005  # absolute, score units
_DATASET_GROWTH_RATIO_THRESHOLD = 1.5    # current_n / baseline_n
_SMOOTHING_WINDOW               = 3      # required successive promotions


def _load_recent_diffs(adapter_dir: Path, n: int) -> List[Dict[str, Any]]:
    """Read the last *n* adapter_diff_*.json reports for *adapter_dir*."""
    reports_dir = adapter_dir / "reports"
    if not reports_dir.is_dir():
        return []
    files = sorted(reports_dir.glob("adapter_diff_*.json"), reverse=True)
    diffs: List[Dict[str, Any]] = []
    for f in files[:n]:
        try:
            diffs.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return diffs


def _load_metadata(adapter_dir: Path) -> Dict[str, Any]:
    meta_path = adapter_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_override(adapter_dir: Path) -> Dict[str, Any]:
    path = adapter_dir / "runtime_recipe_override.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_override(adapter_dir: Path, payload: Dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    path = adapter_dir / "runtime_recipe_override.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _versions_since_last_rank_bump(history: List[Dict[str, Any]]) -> int:
    """
    Count of successful promotions since the last rank change. When the rank
    has never been bumped, this is the total number of promotions ever.
    """
    if not history:
        return 0
    current_r = history[-1].get("lora_r")
    count = 0
    for h in reversed(history):
        if h.get("lora_r") != current_r:
            break
        count += 1
    return count


def _samples_at_last_rank_bump(history: List[Dict[str, Any]]) -> int:
    """
    Walk history to find the sample count *at* the most recent rank change
    (or initial training when no bump has occurred). This is the baseline
    the current dataset must exceed by `_DATASET_GROWTH_RATIO_THRESHOLD`.
    """
    if not history:
        return 0
    current_r = history[-1].get("lora_r")
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("lora_r") != current_r:
            # idx is the entry BEFORE the bump; the bump entry itself is idx+1
            return history[idx + 1].get("n_samples", 0)
    # No transition — current rank is the initial rank; use oldest entry
    return history[0].get("n_samples", 0)


def _effective_current_rank(
    history: List[Dict[str, Any]],
    fallback_r: int,
    fallback_alpha: int,
) -> Tuple[int, int]:
    """
    Resolve the rank/alpha actually present in the saved adapter weights.
    The most recent history entry's `lora_r` is ground truth; callers passing
    the YAML recipe default into *fallback_r* would otherwise produce
    double-bumps once auto-promotion has fired.
    """
    if not history:
        return fallback_r, fallback_alpha
    last = history[-1]
    r = int(last.get("lora_r") or fallback_r)
    a = int(last.get("lora_alpha") or fallback_alpha)
    return r, a


def _check_plateau_signals(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Evaluate the three plateau signals over the smoothing window. Returns a
    summary dict when ALL signals fire (caller should act on it); None when
    any signal is still active or data is insufficient.
    """
    meta = _load_metadata(adapter_dir)
    history = meta.get("history") or []
    diffs = _load_recent_diffs(adapter_dir, _SMOOTHING_WINDOW)

    if len(diffs) < _SMOOTHING_WINDOW:
        log.info(
            "rank_policy: only %s/%s adapter_diff reports for %s — insufficient signal",
            len(diffs), _SMOOTHING_WINDOW, adapter_dir.name,
        )
        return None

    # Signal 1: loss-reduction-per-round
    loss_reductions: List[float] = []
    for d in diffs:
        training = d.get("training") or {}
        prev = training.get("prev_final_loss")
        new  = training.get("new_final_loss")
        if prev is None or new is None or prev <= 0:
            continue
        loss_reductions.append((prev - new) / prev)
    if (
        len(loss_reductions) < _SMOOTHING_WINDOW
        or any(r >= _LOSS_REDUCTION_THRESHOLD for r in loss_reductions)
    ):
        log.info(
            "rank_policy: loss still descending for %s (per-round reductions: %s, threshold=%s)",
            adapter_dir.name,
            [round(x, 4) for x in loss_reductions],
            _LOSS_REDUCTION_THRESHOLD,
        )
        return None

    # Signal 2: regression delta compressed
    reg_deltas: List[float] = []
    for d in diffs:
        reg = d.get("regression") or {}
        delta = reg.get("delta")
        if delta is None:
            continue
        reg_deltas.append(float(delta))
    if (
        len(reg_deltas) < _SMOOTHING_WINDOW
        or any(abs(d) >= _REGRESSION_DELTA_THRESHOLD for d in reg_deltas)
    ):
        log.info(
            "rank_policy: regression delta still meaningful for %s (deltas: %s, threshold=%s)",
            adapter_dir.name,
            [round(x, 4) for x in reg_deltas],
            _REGRESSION_DELTA_THRESHOLD,
        )
        return None

    # Signal 3: dataset growth since last bump
    current_n  = history[-1].get("n_samples", 0) if history else 0
    baseline_n = _samples_at_last_rank_bump(history)
    if baseline_n <= 0:
        log.info("rank_policy: cannot determine dataset baseline for %s", adapter_dir.name)
        return None
    growth_ratio = current_n / baseline_n
    if growth_ratio < _DATASET_GROWTH_RATIO_THRESHOLD:
        log.info(
            "rank_policy: dataset growth insufficient for %s (current=%s baseline=%s ratio=%.2f < %.2f)",
            adapter_dir.name, current_n, baseline_n, growth_ratio,
            _DATASET_GROWTH_RATIO_THRESHOLD,
        )
        return None

    return {
        "loss_reductions":      [round(x, 6) for x in loss_reductions],
        "regression_deltas":    [round(x, 6) for x in reg_deltas],
        "dataset_growth_ratio": round(growth_ratio, 3),
        "current_n_samples":    current_n,
        "baseline_n_samples":   baseline_n,
    }


def is_frozen(adapter_dir: Path) -> bool:
    """Return True iff the override file marks this adapter as frozen."""
    return bool(_load_override(adapter_dir).get("frozen"))


def maybe_promote_rank(
    adapter_dir: Path,
    current_r: int,
    current_alpha: int,
    max_r_cap: int,
    rank_promotion_cooldown: int,
) -> Optional[Dict[str, Any]]:
    """
    Decide whether to bump rank for the adapter rooted at *adapter_dir*.
    On promotion, write/merge into ``runtime_recipe_override.json`` and
    return the override dict. Otherwise return None.

    *current_r* / *current_alpha* are recipe values (typically YAML defaults
    before any override is applied). The function still uses the saved
    adapter's actual rank — read from metadata.json `history[-1].lora_r` —
    as the ground truth for the cap and cooldown checks, so callers may
    safely pass stale recipe defaults.
    """
    if is_frozen(adapter_dir):
        log.info("rank_policy: %s is frozen — no bump", adapter_dir.name)
        return None

    meta = _load_metadata(adapter_dir)
    history = meta.get("history") or []
    eff_r, eff_alpha = _effective_current_rank(history, current_r, current_alpha)

    if eff_r >= max_r_cap:
        log.info(
            "rank_policy: at cap (r=%s == max_r_cap=%s) for %s — no bump",
            eff_r, max_r_cap, adapter_dir.name,
        )
        return None

    # Cooldown
    versions_since = _versions_since_last_rank_bump(history)
    if versions_since < rank_promotion_cooldown:
        log.info(
            "rank_policy: cooldown active for %s (%s/%s promotions since last bump) — no bump",
            adapter_dir.name, versions_since, rank_promotion_cooldown,
        )
        return None

    signals = _check_plateau_signals(adapter_dir)
    if signals is None:
        return None

    # All three signals fired — promote. Double rank (32→64, 64→128, 128→256)
    # capped at max_r_cap. Preserve the recipe's alpha:r ratio so warmup
    # behaviour and effective LR remain consistent across the bump.
    new_r = min(eff_r * 2, max_r_cap)
    if new_r <= eff_r:
        return None  # rounding pinned to cap; nothing to do
    alpha_ratio = (eff_alpha / eff_r) if eff_r > 0 else 2.0
    new_alpha = max(1, int(round(new_r * alpha_ratio)))

    existing = _load_override(adapter_dir)
    payload = {
        **existing,
        "r":                    new_r,
        "alpha":                new_alpha,
        "promoted_from_r":      eff_r,
        "promoted_from_alpha":  eff_alpha,
        "promoted_at_n_samples": signals["current_n_samples"],
        "promoted_at":          datetime.now(timezone.utc).isoformat(),
        "reason":               "auto_rank_promotion_plateau_signal",
        "signal_summary": {
            **signals,
            "versions_since_last_bump": versions_since,
        },
    }
    try:
        _write_override(adapter_dir, payload)
        log.info(
            "rank_policy: PROMOTE %s r=%s→%s alpha=%s→%s (loss_reductions=%s reg_deltas=%s growth=%.2fx)",
            adapter_dir.name, eff_r, new_r, eff_alpha, new_alpha,
            signals["loss_reductions"], signals["regression_deltas"],
            signals["dataset_growth_ratio"],
        )
        return payload
    except OSError as exc:
        log.error("rank_policy: failed to write override for %s: %s", adapter_dir.name, exc)
        return None


def maybe_freeze_at_cap(
    adapter_dir: Path,
    current_r: int,
    current_alpha: int,
    max_r_cap: int,
) -> Optional[Dict[str, Any]]:
    """
    When the adapter is *at* the rank cap AND the same three plateau signals
    fire, mark it frozen. trainer_wrapper.py reads ``is_frozen`` to skip
    further retrains, leaving the adapter idle until an operator merges it
    into a per-agent standalone model.

    Returns the updated override payload on freeze, None otherwise. Uses the
    saved adapter's actual rank (history[-1].lora_r) as ground truth, same
    as ``maybe_promote_rank``.
    """
    if is_frozen(adapter_dir):
        return None

    meta = _load_metadata(adapter_dir)
    history = meta.get("history") or []
    eff_r, _eff_alpha = _effective_current_rank(history, current_r, current_alpha)

    if eff_r < max_r_cap:
        return None  # still has headroom to grow

    signals = _check_plateau_signals(adapter_dir)
    if signals is None:
        return None

    existing = _load_override(adapter_dir)
    payload = {
        **existing,
        "frozen":                True,
        "frozen_at":             datetime.now(timezone.utc).isoformat(),
        "frozen_at_r":           eff_r,
        "freeze_reason":         "at_cap_and_plateau",
        "freeze_signal_summary": signals,
    }
    try:
        _write_override(adapter_dir, payload)
        log.info(
            "rank_policy: FROZEN %s at r=%s (cap=%s) — plateau confirmed, ready for merge-to-base",
            adapter_dir.name, eff_r, max_r_cap,
        )
        return payload
    except OSError as exc:
        log.error("rank_policy: failed to write freeze marker for %s: %s", adapter_dir.name, exc)
        return None


def apply_override(adapter_dir: Path) -> Dict[str, Any]:
    """
    Read runtime_recipe_override.json (if any) and return a dict suitable for
    AdapterRecipe.model_copy(update=...). Returns {} when no override exists
    or it is unreadable. Only ``r`` and ``alpha`` are promotable at runtime
    — every other recipe field stays YAML-driven. The ``frozen`` flag is
    surfaced via ``is_frozen``, not merged into the recipe.
    """
    payload = _load_override(adapter_dir)
    return {k: payload[k] for k in ("r", "alpha") if k in payload}
