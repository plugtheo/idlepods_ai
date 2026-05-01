#!/usr/bin/env python3
"""
Generate .env.vllm from models.yaml so Docker Compose can interpolate
${VLLM_MODEL_ID} and ${VLLM_MAX_MODEL_LEN} in the vllm-primary service.

Usage:
    python scripts/render_compose_env.py [--models-yaml path/to/models.yaml]

Then start with:
    docker compose --env-file .env.vllm up
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts.models import load_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate .env.vllm from models.yaml")
    parser.add_argument(
        "--models-yaml",
        default="models.yaml",
        help="Path to models.yaml (default: models.yaml)",
    )
    parser.add_argument(
        "--output",
        default=".env.vllm",
        help="Output env file path (default: .env.vllm)",
    )
    args = parser.parse_args()

    registry = load_registry(args.models_yaml)
    entry = registry.backends[registry.default_backend]

    lines = [
        f"VLLM_MODEL_ID={entry.model_id}",
        f"VLLM_MAX_MODEL_LEN={entry.max_model_len}",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {args.output}:")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
