from __future__ import annotations

import argparse
from pathlib import Path

from urbanflow.monitoring.model_registry import ModelRegistryError, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all production manifest file hashes.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/production/model_manifest.json"))
    args = parser.parse_args()
    try:
        result = verify_manifest(args.manifest)
    except (ModelRegistryError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Manifest verification failed: {error}") from error
    print("Production manifest verified")
    print(result)


if __name__ == "__main__":
    main()
