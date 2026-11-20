from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the previous local production model manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/production/model_manifest.json"))
    args = parser.parse_args()
    previous = args.manifest.with_name(args.manifest.stem + ".previous" + args.manifest.suffix)
    if not previous.exists():
        raise SystemExit(f"No previous manifest is available: {previous}")
    shutil.copy2(previous, args.manifest)
    print("Production manifest rolled back")
    print("Restored:", args.manifest)


if __name__ == "__main__":
    main()
