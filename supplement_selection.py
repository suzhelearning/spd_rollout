#!/usr/bin/env python3
"""Freeze 29 train and 7 validation raw-valid toss episodes beyond the original 700."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_selection import build_supplemental_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = build_supplemental_selection(args.base_selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"planned": selection["planned"], "by_split": selection["by_split"],
                      "selection_digest": selection["selection_digest"],
                      "path": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
