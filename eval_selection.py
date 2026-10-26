"""Select raw source episodes and account for every planned rollout outcome.

The training manifest contains rendered windows/fragments. Its source_episode field
is the only authority for train/validation membership; raw dataset indexes only
validate source identity and provide the original uncut H5 paths.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


_TASKS = frozenset({"hang_mug", "pyramid", "spelling", "toss_in_bin", "unstack",
                    "vowel_consonant_sort"})
FIVE_TASKS = ("hang_mug", "pyramid", "toss_in_bin", "unstack", "vowel_consonant_sort")
_SPLITS = ("train", "val")
_EPISODE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_OUTCOMES = frozenset({"success", "failure", "unsupported", "error", "pending"})

BASE_SELECTION_SHA256 = "2327cdd2c4ef2a786e930fbfcaa58f19f6af82f5ff1ae7c8dbac2d204aacbd5f"
SUPPLEMENT_METHOD = "supplemental_valid_toss"
SUPPLEMENT_COUNTS = {"train": 29, "val": 7}

def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()



def _source_identity(source_id: str) -> tuple[str, str]:
    if not isinstance(source_id, str) or source_id.count("::") != 1:
        raise ValueError(f"Invalid source episode identity: {source_id!r}")
    source_path, episode_id = source_id.split("::")
    path = PurePosixPath(source_path)
    if (not path.is_absolute() or any(part in (".", "..") for part in source_path.split("/"))
            or path.parent.name != "pre-train" or not _EPISODE_ID.fullmatch(path.name)
            or not _EPISODE_ID.fullmatch(episode_id)):
        raise ValueError(f"Unsafe or malformed source episode identity: {source_id!r}")
    return path.name, episode_id


def _indexed_episodes(dataset: Path, task: str) -> set[str]:
    index = json.loads((dataset / "dataset.json").read_text())
    episodes = index.get("episodes")
    if (index.get("schema_version") != 4 or index.get("task_id") != task
            or not isinstance(episodes, list) or not episodes
            or any(not isinstance(item, str) or not _EPISODE_ID.fullmatch(item)
                   for item in episodes) or len(set(episodes)) != len(episodes)):
        raise ValueError(f"Invalid raw dataset index for {task}: {dataset}")
    return set(episodes)


def select_sources(metadata_path: Path, raw_root: Path, tasks: tuple[str, ...],
                   selection_seed: str = "baseline30k-20261023",
                   fraction: float = 0.2) -> list[dict]:
    """Choose ceil(fraction * distinct sources) independently for each task/split.

    The digest commits to the seed, fraction, source counts and selected IDs;
    each returned row also carries its group's distinct source count.
    """
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= _TASKS:
        raise ValueError("Requested tasks must be distinct members of the six-task evaluation")
    if not isinstance(selection_seed, str) or not selection_seed:
        raise ValueError("Selection seed must be a nonempty string")
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Selection fraction must be in (0, 1]")

    contract = json.loads(Path(metadata_path).read_text()).get("dataset_contract")
    if not isinstance(contract, dict):
        raise ValueError("Missing metadata dataset_contract")
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for split in _SPLITS:
        manifest = contract.get(split)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
            raise ValueError(f"Missing metadata source episodes for {split}")
        for fragment in manifest["episodes"]:
            if not isinstance(fragment, dict) or not isinstance(fragment.get("source"), dict):
                raise ValueError(f"Missing source identity in {split} metadata fragment")
            source_id = fragment["source"].get("source_episode")
            task, episode_id = _source_identity(source_id)
            if task in tasks:
                group = groups.setdefault((split, task), {})
                previous = group.setdefault(episode_id, source_id)
                if previous != source_id:
                    raise ValueError(f"Ambiguous source paths for {split}/{task}/{episode_id}")

    root = Path(raw_root).expanduser().resolve()
    selected_groups = {}
    source_counts = {}
    for task in sorted(tasks):
        dataset = root / task
        indexed = _indexed_episodes(dataset, task)
        for split in _SPLITS:
            group = groups.get((split, task), {})
            if not group:
                raise ValueError(f"No source episodes for {split}/{task}")
            for episode_id in group:
                if episode_id not in indexed:
                    raise ValueError(f"Source {split}/{task}/{episode_id} missing from raw dataset index")
                h5 = dataset / "episodes" / f"{episode_id}.h5"
                if not h5.is_file() or not h5.resolve().is_relative_to(dataset.resolve()):
                    raise ValueError(f"Missing or unsafe raw source H5: {h5}")
            source_counts[(split, task)] = len(group)
            selected_groups[(split, task)] = sorted(
                group, key=lambda episode: (
                    hashlib.sha256(f"{selection_seed}|{split}|{task}|{episode}".encode("utf-8")).digest(),
                    episode))[:math.ceil(fraction * len(group))]
        overlap = groups[("train", task)].keys() & groups[("val", task)].keys()
        if overlap:
            raise ValueError(f"train/val source overlap for {task}: {sorted(overlap)[0]}")

    digest_payload = {"seed": selection_seed, "fraction": fraction,
                      "groups": [{"split": split, "task_id": task,
                                  "source_count": source_counts[(split, task)],
                                  "selected": selected_groups[(split, task)]}
                                 for split in _SPLITS for task in sorted(tasks)]}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return [{"split": split, "task_id": task, "episode_id": episode,
             "raw_dataset": str(root / task),
             "raw_h5": str(root / task / "episodes" / f"{episode}.h5"),
             "source_id": groups[(split, task)][episode],
             "source_count": source_counts[(split, task)], "selection_fraction": fraction,
             "selection_digest": digest}
            for split in _SPLITS for task in sorted(tasks)
            for episode in selected_groups[(split, task)]]


def select_fixed_sources(metadata_path: Path, raw_root: Path, tasks: tuple[str, ...],
                         selection_seed: str, train_count: int, val_count: int) -> list[dict]:
    """Select an exact number of distinct raw source episodes per task and split."""
    if (type(train_count) is not int or type(val_count) is not int
            or train_count <= 0 or val_count <= 0):
        raise ValueError("Fixed source quotas must be positive integers")
    candidates = select_sources(metadata_path, raw_root, tasks, selection_seed, fraction=1)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in candidates:
        groups.setdefault((row["split"], row["task_id"]), []).append(row)
    selected, digest_groups = [], []
    for split in _SPLITS:
        for task in sorted(tasks):
            quota = train_count if split == "train" else val_count
            group = groups[(split, task)]
            if len(group) < quota:
                raise ValueError(f"{split}/{task}: quota {quota} exceeds {len(group)} available sources")
            chosen = group[:quota]
            selected.extend(chosen)
            digest_groups.append({"split": split, "task_id": task,
                                  "source_count": len(group),
                                  "selected": [row["episode_id"] for row in chosen]})
    digest = hashlib.sha256(json.dumps(
        {"seed": selection_seed, "method": "fixed_count_per_task_split",
         "train_count": train_count, "val_count": val_count, "groups": digest_groups},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return [{**{key: value for key, value in row.items()
                if key not in ("selection_fraction", "selection_digest")},
             "selection_digest": digest} for row in selected]


def build_supplemental_selection(base_selection_path: Path) -> dict:
    """Extend, never revise, the SHA-pinned 700-source baseline with raw-valid toss sources."""
    # Import locally: baseline_raw_rollout imports this module while initializing.
    from baseline_raw_rollout import _raw_scene

    path = Path(base_selection_path).expanduser().resolve(strict=True)
    base_sha = _file_sha256(path)
    if base_sha != BASE_SELECTION_SHA256:
        raise ValueError("Baseline 700 selection SHA256 differs from the frozen experiment")
    base = json.loads(path.read_text())
    if (base.get("selection_method") != "fixed_count_per_task_split"
            or base.get("planned") != 700 or base.get("tasks") != list(FIVE_TASKS)
            or base.get("counts_per_task_split") != {"train": 100, "val": 40}
            or base.get("by_split") != {"train": 500, "val": 200}
            or "fraction" in base
            or not isinstance(base.get("selection_seed"), str)):
        raise ValueError("Frozen baseline selection is not the original five-task 700")
    source_split = Path(base["source_split"]).expanduser().resolve(strict=True)
    raw_root = Path(base["raw_root"]).expanduser().resolve(strict=True)
    baseline = select_fixed_sources(source_split, raw_root, FIVE_TASKS,
                                    base["selection_seed"], 100, 40)
    if (base["episodes"] != baseline
            or base["selection_digest"] != baseline[0]["selection_digest"]):
        raise ValueError("Frozen baseline 700 rows differ from the raw source split")
    excluded = {row["source_id"] for row in baseline}
    chosen: dict[str, list[dict]] = {split: [] for split in _SPLITS}
    for row in select_sources(source_split, raw_root, ("toss_in_bin",),
                              base["selection_seed"], fraction=1):
        if row["source_id"] in excluded:
            continue
        split = row["split"]
        if len(chosen[split]) >= SUPPLEMENT_COUNTS[split]:
            continue
        try:
            _raw_scene(row)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            # Failed raw H5 header, scene load, or v4 identity; never inspect model scores.
            continue
        chosen[split].append(row)
    if any(len(chosen[split]) != SUPPLEMENT_COUNTS[split] for split in _SPLITS):
        raise ValueError("Not enough distinct raw-valid toss sources beyond the frozen baseline")
    ordered = [row for split in _SPLITS for row in chosen[split]]
    identities = [row["source_id"] for row in ordered]
    if len(set(identities)) != 36 or excluded.intersection(identities):
        raise ValueError("Supplemental toss sources overlap the frozen baseline or another split")
    with_hashes = [{**row, "raw_h5_sha256": _file_sha256(Path(row["raw_h5"]))}
                   for row in ordered]
    digest_payload = {"base_selection_sha256": base_sha, "selection_seed": base["selection_seed"],
                      "selection_method": SUPPLEMENT_METHOD,
                      "by_split": {split: [{"source_id": row["source_id"],
                                           "raw_h5_sha256": row["raw_h5_sha256"]}
                                          for row in with_hashes if row["split"] == split]
                                   for split in _SPLITS}}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    rows = [{**row, "selection_digest": digest} for row in with_hashes]
    return {"selection_method": SUPPLEMENT_METHOD, "base_selection_path": str(path),
            "base_selection_sha256": base_sha, "selection_seed": base["selection_seed"],
            "source_split": str(source_split), "raw_root": str(raw_root),
            "tasks": ["toss_in_bin"], "counts_per_task_split": dict(SUPPLEMENT_COUNTS),
            "by_split": dict(SUPPLEMENT_COUNTS), "planned": 36,
            "selection_digest": digest, "episodes": rows}


def _identity(row: dict) -> tuple[str, str, str]:
    if not isinstance(row, dict):
        raise ValueError("Outcome/plan row must be an object")
    identity = (row.get("split"), row.get("task_id"), row.get("episode_id"))
    if (identity[0] not in _SPLITS or identity[1] not in _TASKS
            or not isinstance(identity[2], str) or not _EPISODE_ID.fullmatch(identity[2])):
        raise ValueError(f"Invalid evaluation identity: {identity!r}")
    return identity


def summarize(results: list[dict], planned: list[dict]) -> dict:
    """Summarize complete explicit outcomes without counting unsupported/error as scored.

    An unfinished rollout must be present with status='pending'; silently absent
    results, duplicate rows and results not in the selection are errors.
    """
    identities = [_identity(row) for row in planned]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate planned source identity")
    expected = set(identities)
    counts: dict[str, dict[str, Counter]] = {split: {} for split in _SPLITS}
    for split, task, _ in identities:
        if task not in counts[split]:
            counts[split][task] = Counter()
        counts[split][task]["planned"] += 1
    observed = set()
    for row in results:
        identity = _identity(row)
        if identity not in expected:
            raise ValueError(f"Unplanned source outcome: {identity}")
        if identity in observed:
            raise ValueError(f"Duplicate source outcome: {identity}")
        observed.add(identity)
        status = row.get("status")
        if status not in _OUTCOMES:
            raise ValueError(f"Invalid source outcome status: {status!r}")
        counts[identity[0]][identity[1]][status] += 1
    if observed != expected:
        raise ValueError(f"Missing results; mark {len(expected - observed)} as pending explicitly")
    summary = {split: {} for split in _SPLITS}
    for split, tasks in counts.items():
        for task, count in tasks.items():
            scored = count["success"] + count["failure"]
            summary[split][task] = {
                "planned": count["planned"],
                **{status: count[status] for status in
                   ("success", "failure", "unsupported", "error", "pending")},
                "scored": scored, "success_rate": count["success"] / scored if scored else None,
            }
    return summary
