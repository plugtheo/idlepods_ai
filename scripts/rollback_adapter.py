"""
Roll back a LoRA adapter to its previous version.

The rollback API groups adapters by backend (from models.yaml), not by capability.
Use --backend to specify which backend's active adapter to roll back.
--capability is accepted for CLI convenience but is not sent to the API.

Usage:
    python scripts/rollback_adapter.py --backend primary
    python scripts/rollback_adapter.py --backend primary --inference-url http://localhost:8010
    python scripts/rollback_adapter.py --backend primary --auto --reason "auto_fallback_threshold"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx


def _default_backend() -> str:
    try:
        from shared.contracts.models import load_registry
        return load_registry().default_backend
    except Exception:
        return "primary"


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll back an adapter to its previous version")
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name from models.yaml (default: registry default_backend)",
    )
    parser.add_argument(
        "--capability",
        default=None,
        help="Capability label (informational only — backend is resolved from registry)",
    )
    parser.add_argument("--inference-url", default="http://localhost:8010",
                        help="Inference service base URL")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: skip interactive confirmation prompt (used by auto-rollback path)",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Free-text reason recorded into the manifest history (e.g. 'auto_fallback_threshold')",
    )
    args = parser.parse_args()

    backend = args.backend or _default_backend()

    if not args.auto:
        try:
            confirm = input(
                f"Roll back adapter for backend '{backend}'? This cannot be undone. [y/N] "
            ).strip().lower()
        except EOFError:
            confirm = ""
        if confirm != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    payload: dict = {"backend": backend}
    if args.reason:
        payload["reason"] = args.reason

    try:
        resp = httpx.post(
            f"{args.inference_url}/adapters/rollback",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"[OK] rolled_back={data.get('rolled_back')}  previous_path={data.get('previous_path')}")
    except httpx.HTTPStatusError as exc:
        print(f"[FAIL] HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
