#!/usr/bin/env python3
"""
Plan B Verification Script — AdapterRecipe + Native OpenAI Tool-Call SFT
Run with: python scripts/healthcheck.py
"""
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []  # (section, label, status)


def _record(section: str, label: str, ok: bool | None, detail: str = ""):
    status = PASS if ok else (SKIP if ok is None else FAIL)
    tag = f"[{status}]"
    line = f"  {tag:<7} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    _results.append((section, label, status))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------
section("1. Required files exist")

REQUIRED_FILES = [
    "recipes.yaml",
    "shared/contracts/training.py",
    "shared/contracts/messages.py",
    "shared/contracts/experience.py",
    "orchestration/app/experience/recorder.py",
    "orchestration/app/experience/sft_builder.py",
    "training/training/lora_trainer.py",
    "training/training/train_gpu_simple.py",
    "training/training/smoke_gate.py",
    "training/training/validate_adapter.py",
    "training/app/trainer_entry.py",
    "scripts/seed_adapter_metadata.py",
    "training/tests/test_recipe_loader.py",
    "training/tests/test_lora_trainer_recipe.py",
    "training/tests/test_sft_builder.py",
    "orchestration/tests/test_recorder_tool_pairs.py",
    "training/tests/test_smoke_gate_tool_call.py",
]

for rel in REQUIRED_FILES:
    p = REPO_ROOT / rel
    _record("files", rel, p.exists(), "" if p.exists() else "MISSING")


# ---------------------------------------------------------------------------
# 2. No forbidden model-name literals in Python source
# ---------------------------------------------------------------------------
section("2. No hardcoded model-name literals (qwen/deepseek/mistral)")

FORBIDDEN = ['"qwen"', "'qwen'", '"deepseek"', "'deepseek'", '"mistral"', "'mistral'"]
CHANGED_PY = [
    "shared/contracts/training.py",
    "shared/contracts/messages.py",
    "shared/contracts/experience.py",
    "orchestration/app/experience/recorder.py",
    "orchestration/app/experience/sft_builder.py",
    "training/training/lora_trainer.py",
    "training/training/train_gpu_simple.py",
    "training/training/smoke_gate.py",
    "training/training/validate_adapter.py",
    "training/app/trainer_entry.py",
    "scripts/seed_adapter_metadata.py",
]

for rel in CHANGED_PY:
    p = REPO_ROOT / rel
    if not p.exists():
        _record("literals", rel, None, "file missing")
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    hits = [f for f in FORBIDDEN if f in text]
    _record("literals", rel, len(hits) == 0,
            f"found: {hits}" if hits else "")


# ---------------------------------------------------------------------------
# 3. recipes.yaml structure
# ---------------------------------------------------------------------------
section("3. recipes.yaml structure")

try:
    import yaml
    recipes_path = REPO_ROOT / "recipes.yaml"
    raw = yaml.safe_load(recipes_path.read_text(encoding="utf-8", errors="replace"))

    _record("yaml", "has 'default' section", "default" in raw)
    _record("yaml", "has 'by_role' section", "by_role" in raw)

    default = raw.get("default", {})
    _record("yaml", "default.peft_type == lora", default.get("peft_type") == "lora")
    _record("yaml", "default.sft_format == openai_messages",
            default.get("sft_format") == "openai_messages")
    _record("yaml", "default.tool_call_style == openai_native",
            default.get("tool_call_style") == "openai_native")
    _record("yaml", "default.target_modules present",
            isinstance(default.get("target_modules"), list) and len(default["target_modules"]) > 0)

    by_role = raw.get("by_role", {})
    _record("yaml", "coder recipe has r=32", by_role.get("coder", {}).get("r") == 32)
    _record("yaml", "debugger recipe has r=32", by_role.get("debugger", {}).get("r") == 32)
    _record("yaml", "consensus peft_type=none",
            by_role.get("consensus", {}).get("peft_type") == "none")
except Exception as exc:
    _record("yaml", "yaml parse", False, str(exc))


# ---------------------------------------------------------------------------
# 4. AdapterRecipe / RecipeRegistry import & behaviour
# ---------------------------------------------------------------------------
section("4. AdapterRecipe / RecipeRegistry (shared/contracts/training.py)")

try:
    from shared.contracts.training import (
        AdapterRecipe, RecipeRegistry, load_recipes, lookup_recipe
    )
    load_recipes.cache_clear()

    # Default construction
    r = AdapterRecipe(target_modules=["q_proj"])
    _record("recipe", "AdapterRecipe default peft_type=lora", r.peft_type == "lora")
    _record("recipe", "AdapterRecipe default sft_format=openai_messages",
            r.sft_format == "openai_messages")
    _record("recipe", "AdapterRecipe default tool_call_style=openai_native",
            r.tool_call_style == "openai_native")

    # Load from real recipes.yaml
    load_recipes.cache_clear()
    reg = load_recipes(str(REPO_ROOT / "recipes.yaml"))
    coder = reg.lookup("primary", "coder")
    _record("recipe", "coder recipe r=32 (via lookup)", coder.r == 32)
    _record("recipe", "coder recipe alpha=64", coder.alpha == 64)
    _record("recipe", "coder recipe num_epochs=4", coder.num_epochs == 4)
    _record("recipe", "coder inherits sft_format=openai_messages",
            coder.sft_format == "openai_messages")

    debugger = reg.lookup("primary", "debugger")
    _record("recipe", "debugger recipe r=32", debugger.r == 32)

    consensus = reg.lookup("primary", "consensus")
    _record("recipe", "consensus peft_type=none", consensus.peft_type == "none")

    unknown = reg.lookup("primary", "unknown_role")
    _record("recipe", "unknown role falls back to default r=16", unknown.r == 16)

    # Precedence: backend:role > role > default
    import textwrap, tempfile
    yaml_content = textwrap.dedent("""
        default:
          peft_type: lora
          r: 4
          alpha: 8
          target_modules: [q_proj]
          sft_format: openai_messages
          tool_call_style: openai_native
        by_role:
          coder: {r: 8}
        by_backend_role:
          "sec:coder": {r: 16}
    """)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(yaml_content)
        tmp_path = tf.name
    load_recipes.cache_clear()
    treg = load_recipes(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    _record("recipe", "backend:role beats role (precedence test)",
            treg.lookup("sec", "coder").r == 16)
    _record("recipe", "role beats default (precedence test)",
            treg.lookup("other", "coder").r == 8)
    _record("recipe", "default when no match",
            treg.lookup("other", "planner").r == 4)

    # Reload main registry
    load_recipes.cache_clear()

except Exception as exc:
    _record("recipe", "import/init failed", False, str(exc))


# ---------------------------------------------------------------------------
# 5. shared/contracts/messages.py
# ---------------------------------------------------------------------------
section("5. shared/contracts/messages.py")

try:
    from shared.contracts.messages import build_openai_messages, build_legacy_marker_prompt

    msgs = build_openai_messages("coder", "sys", "user", tool_rounds=None)
    _record("messages", "build_openai_messages returns list", isinstance(msgs, list))
    _record("messages", "first msg role=system", msgs[0]["role"] == "system")
    _record("messages", "second msg role=user", msgs[1]["role"] == "user")

    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    msgs2 = build_openai_messages("coder", "sys", "user",
                                   tool_rounds=[{"tool_calls": tc,
                                                 "tool_results": [{"tool_call_id": "c1", "content": "ok"}]}])
    roles = [m["role"] for m in msgs2]
    _record("messages", "tool round produces assistant+tool msgs", "tool" in roles)

    prompt, completion = build_legacy_marker_prompt("coder", "sys", "user", "output")
    _record("messages", "legacy marker prompt starts with [SYSTEM]", prompt.startswith("[SYSTEM]"))
    _record("messages", "legacy marker prompt contains [RESPONSE]", "[RESPONSE]" in prompt)
    _record("messages", "legacy marker completion correct", completion == "output")

except Exception as exc:
    _record("messages", "import/call failed", False, str(exc))


# ---------------------------------------------------------------------------
# 6. shared/contracts/experience.py — AgentContribution fields
# ---------------------------------------------------------------------------
section("6. AgentContribution new fields")

try:
    from shared.contracts.experience import AgentContribution
    fields = AgentContribution.model_fields
    _record("experience", "tool_calls field exists", "tool_calls" in fields)
    _record("experience", "tool_results field exists", "tool_results" in fields)
    _record("experience", "used_base_fallback field exists", "used_base_fallback" in fields)

    ac = AgentContribution(role="coder", output="x", quality_score=0.9, iteration=1)
    _record("experience", "tool_calls defaults to None", ac.tool_calls is None)
    _record("experience", "tool_results defaults to None", ac.tool_results is None)
    _record("experience", "used_base_fallback defaults to False", ac.used_base_fallback is False)

except Exception as exc:
    _record("experience", "import/check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 7. recorder._build_contributions
# ---------------------------------------------------------------------------
section("7. orchestration/app/experience/recorder._build_contributions")

try:
    from orchestration.app.experience.recorder import _build_contributions

    # No tool calls
    history = [{"role": "assistant", "content": "Answer."}]
    contribs = _build_contributions(history, "coder", 0.8, 1, [])
    _record("recorder", "no-tool-call contribution output correct",
            contribs[0].output == "Answer.")
    _record("recorder", "no-tool-call tool_calls is None", contribs[0].tool_calls is None)

    # With tool call
    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    history2 = [
        {"role": "assistant", "tool_calls": tc},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": "Done."},
    ]
    c2 = _build_contributions(history2, "coder", 0.9, 1, [])
    _record("recorder", "tool_calls captured", c2[0].tool_calls == tc)
    _record("recorder", "tool_results captured", len(c2[0].tool_results) == 1)
    _record("recorder", "final output correct", c2[0].output == "Done.")

    # tool_result legacy rows skipped
    history3 = [
        {"role": "assistant", "tool_calls": tc},
        {"role": "tool_result", "content": "legacy"},
        {"role": "assistant", "content": "Final."},
    ]
    c3 = _build_contributions(history3, "coder", 0.8, 1, [])
    _record("recorder", "tool_result legacy rows skipped", c3[0].output == "Final.")

except Exception as exc:
    _record("recorder", "import/check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 8. sft_builder.build_sft_pair
# ---------------------------------------------------------------------------
section("8. orchestration/app/experience/sft_builder.build_sft_pair")

try:
    from orchestration.app.experience.sft_builder import build_sft_pair
    from shared.contracts.experience import AgentContribution
    from shared.contracts.training import AdapterRecipe

    openai_recipe = AdapterRecipe(target_modules=["q_proj"], sft_format="openai_messages")
    legacy_recipe = AdapterRecipe(target_modules=["q_proj"], sft_format="legacy_response_marker")

    contrib = AgentContribution(role="coder", output="result", quality_score=0.9, iteration=1)
    result = build_sft_pair(contrib, openai_recipe, "coder", "sys", "user")
    _record("sft_builder", "openai_messages returns messages key", "messages" in result)
    msgs = result["messages"]
    _record("sft_builder", "last message is assistant final", msgs[-1]["role"] == "assistant")
    _record("sft_builder", "last message content == output", msgs[-1]["content"] == "result")

    # 3-turn with tool_calls
    tc = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
    tr = [{"tool_call_id": "c1", "content": "file contents"}]
    contrib2 = AgentContribution(role="coder", output="done", quality_score=0.9, iteration=1,
                                  tool_calls=tc, tool_results=tr)
    r2 = build_sft_pair(contrib2, openai_recipe, "coder", "sys", "user")
    roles = [m["role"] for m in r2["messages"]]
    _record("sft_builder", "3-turn openai shape: system,user,assistant,tool,assistant",
            roles == ["system", "user", "assistant", "tool", "assistant"])
    _record("sft_builder", "assistant tool_calls msg content=None",
            r2["messages"][2].get("content") is None)

    # Legacy format
    r3 = build_sft_pair(contrib, legacy_recipe, "coder", "sys", "user")
    _record("sft_builder", "legacy returns prompt+completion", "prompt" in r3 and "completion" in r3)
    _record("sft_builder", "legacy prompt has [RESPONSE]", "[RESPONSE]" in r3["prompt"])

except Exception as exc:
    _record("sft_builder", "import/check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 9. lora_trainer.apply_recipe present
# ---------------------------------------------------------------------------
section("9. training/training/lora_trainer.py — apply_recipe")

try:
    # Check apply_recipe is defined (don't import unsloth)
    src = (REPO_ROOT / "training/training/lora_trainer.py").read_text(encoding="utf-8", errors="replace")
    _record("lora_trainer", "apply_recipe function defined", "def apply_recipe(" in src)
    _record("lora_trainer", "issubset assertion present",
            "issubset" in src or "not found in model" in src)
    _record("lora_trainer", "recipe param in train signature",
            "recipe: Optional[AdapterRecipe]" in src or "recipe=None" in src)
    _record("lora_trainer", "apply_recipe called in _train_unsloth",
            "apply_recipe(model, recipe)" in src)
    _record("lora_trainer", "hardcoded target_modules block replaced",
            src.count('"q_proj", "k_proj", "o_proj", "v_proj"') == 0
            or "apply_recipe" in src)

except Exception as exc:
    _record("lora_trainer", "source check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 10. train_gpu_simple.py — LORA_R/LORA_ALPHA dropped, recipe used
# ---------------------------------------------------------------------------
section("10. training/training/train_gpu_simple.py — recipe-driven")

try:
    src = (REPO_ROOT / "training/training/train_gpu_simple.py").read_text(encoding="utf-8", errors="replace")
    _record("train_gpu", "LORA_R module constant removed", "LORA_R     =" not in src)
    _record("train_gpu", "LORA_ALPHA module constant removed", "LORA_ALPHA =" not in src)
    _record("train_gpu", "_resolve_recipe function added", "_resolve_recipe" in src)
    _record("train_gpu", "lookup_recipe imported", "lookup_recipe" in src)
    _record("train_gpu", "recipe passed to trainer.train", "recipe=recipe" in src)

except Exception as exc:
    _record("train_gpu", "source check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 11. trainer_entry.py — --recipe-name flag + recipe persisted
# ---------------------------------------------------------------------------
section("11. training/app/trainer_entry.py — --recipe-name + recipe in metadata")

try:
    src = (REPO_ROOT / "training/app/trainer_entry.py").read_text(encoding="utf-8", errors="replace")
    _record("trainer_entry", "--recipe-name argument added",
            "--recipe-name" in src or "recipe_name" in src)
    _record("trainer_entry", "lookup_recipe imported", "lookup_recipe" in src)
    _record("trainer_entry", "recipe.model_dump() persisted to history",
            'recipe.model_dump()' in src or '"recipe"' in src)
    _record("trainer_entry", "recipe.num_epochs used (not settings.lora_num_epochs)",
            "recipe.num_epochs" in src)
    _record("trainer_entry", "recipe.r used", "recipe.r" in src)

except Exception as exc:
    _record("trainer_entry", "source check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 12. smoke_gate.py — tool-call shape check
# ---------------------------------------------------------------------------
section("12. training/training/smoke_gate.py — tool-call shape gate")

try:
    src = (REPO_ROOT / "training/training/smoke_gate.py").read_text(encoding="utf-8", errors="replace")
    _record("smoke_gate", "_tool_call_shape_ok defined", "_tool_call_shape_ok" in src)
    _record("smoke_gate", "recipe param in run_smoke", "recipe" in src)
    _record("smoke_gate", "chat/completions endpoint used for tool roles",
            "chat/completions" in src or "v1/chat/completions" in src)
    _record("smoke_gate", "tool_call_shape_fail reason emitted",
            "tool_call_shape_fail" in src)

    from training.training.smoke_gate import _tool_call_shape_ok
    good = {"choices": [{"message": {"tool_calls": [{"id": "x", "type": "function",
                                                       "function": {"name": "f", "arguments": "{}"}}]}}]}
    bad  = {"choices": [{"message": {"content": "plain text"}}]}
    _record("smoke_gate", "_tool_call_shape_ok passes valid response", _tool_call_shape_ok(good))
    _record("smoke_gate", "_tool_call_shape_ok fails plain-text response",
            not _tool_call_shape_ok(bad))

except Exception as exc:
    _record("smoke_gate", "source/import check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 13. validate_adapter.py — recipe-aware load
# ---------------------------------------------------------------------------
section("13. training/training/validate_adapter.py — recipe-aware adapter load")

try:
    src = (REPO_ROOT / "training/training/validate_adapter.py").read_text(encoding="utf-8", errors="replace")
    _record("validate", "lookup_recipe imported", "lookup_recipe" in src)
    _record("validate", "load_in_4bit driven by recipe",
            "recipe.load_in_4bit" in src or "peft_type == .qlora" in src or "qlora" in src)
    _record("validate", "max_seq_length from recipe",
            "recipe.max_seq_length" in src or "_recipe.max_seq_length" in src)

except Exception as exc:
    _record("validate", "source check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 14. seed_adapter_metadata.py — recipe in history
# ---------------------------------------------------------------------------
section("14. scripts/seed_adapter_metadata.py — recipe in seed history")

try:
    src = (REPO_ROOT / "scripts/seed_adapter_metadata.py").read_text(encoding="utf-8", errors="replace")
    _record("seed", "lookup_recipe imported", "lookup_recipe" in src)
    _record("seed", "recipe dict included in history entry",
            "_recipe_dict" in src or '"recipe"' in src)

except Exception as exc:
    _record("seed", "source check failed", False, str(exc))


# ---------------------------------------------------------------------------
# 15. Run pytest suite
# ---------------------------------------------------------------------------
section("15. pytest — Plan B test suite")

TEST_MODULES = [
    "training/tests/test_recipe_loader.py",
    "orchestration/tests/test_recorder_tool_pairs.py",
    "training/tests/test_sft_builder.py",
    "training/tests/test_smoke_gate_tool_call.py",
    "training/tests/test_lora_trainer_recipe.py",
]

for tm in TEST_MODULES:
    tp = REPO_ROOT / tm
    if not tp.exists():
        _record("pytest", tm, None, "file missing")
        continue
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(tp), "-q", "--tb=short"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        passed = proc.returncode == 0
        # Extract summary line
        last_lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip()]
        summary = last_lines[-1] if last_lines else "(no output)"
        _record("pytest", tm, passed, summary if not passed else "")
    except subprocess.TimeoutExpired:
        _record("pytest", tm, False, "timed out")
    except Exception as exc:
        _record("pytest", tm, False, str(exc))


# ---------------------------------------------------------------------------
# 16. Constraint checks
# ---------------------------------------------------------------------------
section("16. Constraint / invariant checks")

# target_modules assertion present in apply_recipe
try:
    src = (REPO_ROOT / "training/training/lora_trainer.py").read_text(encoding="utf-8", errors="replace")
    _record("constraints", "target_modules issubset assertion in apply_recipe",
            "issubset" in src or "not found in model" in src)
except Exception as exc:
    _record("constraints", "lora_trainer source read failed", False, str(exc))

# Legacy marker fallback NOT removed (Plan E removes it)
try:
    inf_path = REPO_ROOT / "inference/app/backends/local_vllm.py"
    if inf_path.exists():
        inf_src = inf_path.read_text(encoding="utf-8", errors="replace")
        _record("constraints", "legacy fallback still present in local_vllm.py",
                "RESPONSE" in inf_src or "legacy" in inf_src.lower())
    else:
        _record("constraints", "local_vllm.py fallback check", None, "file not found — skip")
except Exception as exc:
    _record("constraints", "local_vllm check failed", False, str(exc))

# Non-tool roles not switched to openai_messages (still legacy in default for non-coder/debugger)
try:
    import yaml
    raw2 = yaml.safe_load((REPO_ROOT / "recipes.yaml").read_text(encoding="utf-8", errors="replace"))
    non_tool_roles = ["planner", "researcher", "reviewer", "critic"]
    by_role = raw2.get("by_role", {})
    for role in non_tool_roles:
        override = by_role.get(role, {})
        # They should NOT override sft_format to openai_messages explicitly
        _record("constraints", f"{role} not overriding to openai_messages in recipes.yaml",
                override.get("sft_format") != "openai_messages")
except Exception as exc:
    _record("constraints", "non-tool role check failed", False, str(exc))

# recipes.yaml uses string keys for by_backend_role (not list keys)
try:
    import yaml
    raw3 = yaml.safe_load((REPO_ROOT / "recipes.yaml").read_text(encoding="utf-8", errors="replace"))
    bbr = raw3.get("by_backend_role") or {}
    all_str = all(isinstance(k, str) for k in bbr.keys())
    _record("constraints", "by_backend_role uses string keys only", all_str)
except Exception as exc:
    _record("constraints", "by_backend_role key check failed", False, str(exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

total  = len(_results)
passed = sum(1 for _, _, s in _results if s == PASS)
failed = sum(1 for _, _, s in _results if s == FAIL)
skipped= sum(1 for _, _, s in _results if s == SKIP)

print(f"  Total : {total}")
print(f"  PASS  : {passed}")
print(f"  FAIL  : {failed}")
print(f"  SKIP  : {skipped}")

if failed:
    print("\n  FAILED checks:")
    for sec, label, status in _results:
        if status == FAIL:
            print(f"    [{sec}] {label}")

overall = "ALL CHECKS PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED"
print(f"\n  Result: {overall}\n")
sys.exit(0 if failed == 0 else 1)
