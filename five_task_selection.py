#!/usr/bin/env python3
"""Freeze five-task 100/40 source episodes from the baseline training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_selection import FIVE_TASKS, select_fixed_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="baseline30k-five100val40-20261024")
    args = parser.parse_args()
    source_split = args.source_split.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    rows = select_fixed_sources(source_split, raw_root, FIVE_TASKS, args.seed, 100, 40)
    if len(rows) != 700:
        raise RuntimeError("Five tasks must yield exactly 500 train and 200 val sources")
    selection = {"model": "baseline_30000.pt", "source_split": str(source_split),
                 "raw_root": str(raw_root), "selection_method": "fixed_count_per_task_split",
                 "selection_seed": args.seed, "tasks": list(FIVE_TASKS),
                 "counts_per_task_split": {"train": 100, "val": 40},
                 "by_split": {"train": 500, "val": 200}, "planned": len(rows),
                 "selection_digest": rows[0]["selection_digest"], "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"planned": len(rows), "by_split": selection["by_split"],
                      "selection_digest": selection["selection_digest"],
                      "path": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
