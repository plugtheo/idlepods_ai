"""
Roll back a LoRA adapter to its previous version.

Usage:
    python scripts/rollback_adapter.py --capability coding
    python scripts/rollback_adapter.py --capability coding --inference-url http://localhost:8010
"""
import argparse
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll back an adapter to its previous version")
    parser.add_argument("--capability", required=True,
                        help="Capability to roll back (coding, debugging, review, planning, research, criticism)")
    parser.add_argument("--inference-url", default="http://localhost:8010",
                        help="Inference service base URL")
    args = parser.parse_args()

    try:
        resp = httpx.post(
            f"{args.inference_url}/adapters/rollback",
            json={"capability": args.capability},
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
