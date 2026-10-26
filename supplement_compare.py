#!/usr/bin/env python3
"""Compare two independently audited 700-source and 36-source GPU runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from aggregate_evaluation import model_identity
from eval_selection import FIVE_TASKS, build_supplemental_selection
from five_task_audit import _FROZEN_BASE_SELECTION_SHA256, result_evidence_sha256
from rollout_policy import save_json

_FIELDS = ("planned", "success", "failure", "unsupported", "error", "pending", "scored")
_STATUSES = ("success", "failure", "unsupported", "error")


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def _sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _counts(values: dict) -> dict:
    counts = {key: values[key] for key in _FIELDS}
    if (any(type(value) is not int or value < 0 for value in counts.values())
            or counts["planned"] != sum(counts[key] for key in _STATUSES) + counts["pending"]
            or counts["scored"] != counts["success"] + counts["failure"]):
        raise ValueError("Invalid planned/scored outcome partition")
    return {**counts, "success_per_planned": counts["success"] / counts["planned"] if counts["planned"] else None,
            "success_per_scored": counts["success"] / counts["scored"] if counts["scored"] else None}


def _sum_counts(*items: dict) -> dict:
    return _counts({field: sum(item[field] for item in items) for field in _FIELDS})


def _old_evidence(paths: dict, model: str, base: dict, base_sha: str) -> tuple[dict, dict, dict]:
    final_path, audit_path, aggregate_path = (Path(paths[key]) for key in ("old_final", "old_audit", "old_aggregate"))
    final, audit, aggregate = (_read(path) for path in (final_path, audit_path, aggregate_path))
    if (final.get("status") != "final" or final.get("issues")
            or audit.get("status") != "passed" or audit.get("failures")
            or audit.get("planned") != 700 or audit.get("terminal") != 700
            or audit.get("known_invalid_raw") != 36):
        raise ValueError(f"{model} old 700 audit/final not passed")
    if not isinstance(final.get("generated_at"), str) or not final["generated_at"]:
        raise ValueError(f"{model} historical final generation time missing")
    if (aggregate.get("selection_sha256") != base_sha or audit.get("selection_sha256") != base_sha
            or final.get("selection_digest") != base.get("selection_digest")
            or audit.get("selection_digest") != base.get("selection_digest")
            or aggregate.get("selection_digest") != base.get("selection_digest")
            or audit.get("aggregate_sha256") != _sha(aggregate_path)
            or final.get("inputs", {}).get("audit", {}).get("sha256") != _sha(audit_path)
            or final.get("inputs", {}).get("aggregate", {}).get("sha256") != _sha(aggregate_path)):
        raise ValueError(f"{model} historical source or audit provenance mismatch")
    if (aggregate.get("status") != "complete" or aggregate.get("method") != "gpu_vector"
            or aggregate.get("control_hz") != 30):
        raise ValueError(f"{model} historical aggregate is not complete GPU evidence")
    expected_model = model_identity(aggregate, legacy_baseline=(model == "baseline"))
    if (expected_model[0] != model
            or any(model_identity(record, legacy_baseline=(model == "baseline")) != expected_model
                   for record in (final, audit))):
        raise ValueError(f"{model} historical model identity mismatch")
    if (not isinstance(audit.get("run_root"), str)
            or not Path(audit["run_root"]).is_absolute()
            or audit.get("result_evidence_sha256") !=
            result_evidence_sha256(base, Path(audit["run_root"]), 8)):
        raise ValueError(f"{model} historical result evidence changed after audit")
    historical = _counts(final["total"])
    if (historical["planned"] != 700 or historical["pending"]
            or historical["error"] != 36
            or any(historical[key] != sum(aggregate["counts_by_split"][split][key]
                                           for split in ("train", "val")) for key in _FIELDS)):
        raise ValueError(f"{model} historical counts mismatch")
    if (set(final.get("counts_by_split", {})) != {"train", "val"}
            or set(final.get("counts_by_task", {})) != {"train", "val"}
            or any(any(final["counts_by_split"][split][key] != aggregate["counts_by_split"][split][key]
                       for key in _FIELDS)
                   or set(final["counts_by_task"][split]) != set(FIVE_TASKS)
                   or any(any(final["counts_by_task"][split][task][key]
                              != aggregate["counts_by_task"][split][task][key] for key in _FIELDS)
                          for task in FIVE_TASKS)
                   for split in ("train", "val"))):
        raise ValueError(f"{model} historical final/counts mismatch")
    for split, quota, bad in (("train", 500, 29), ("val", 200, 7)):
        tasks = aggregate["counts_by_task"][split]
        if (set(tasks) != set(FIVE_TASKS)
                or any(tasks[task]["planned"] != (100 if split == "train" else 40)
                       for task in FIVE_TASKS)
                or tasks["toss_in_bin"]["error"] != bad
                or sum(tasks[task]["error"] for task in FIVE_TASKS if task != "toss_in_bin")
                or _counts(aggregate["counts_by_split"][split])["planned"] != quota
                or any(sum(tasks[task][key] for task in FIVE_TASKS)
                       != aggregate["counts_by_split"][split][key] for key in _FIELDS)
                or any(audit["counts_by_split"][split][key] != aggregate["counts_by_split"][split][key]
                       for key in _FIELDS)):
            raise ValueError(f"{model} historical {split} counts mismatch")
    return final, audit, aggregate


def _new_evidence(paths: dict, model: str, selection: dict, selection_sha: str) -> tuple[dict, dict, dict, dict]:
    aggregate_path, audit_path, launch_path = (Path(paths[key]) for key in
                                               ("new_aggregate", "new_audit", "new_launch"))
    aggregate, audit, launch = (_read(path) for path in (aggregate_path, audit_path, launch_path))
    if (audit.get("status") != "passed" or audit.get("failures")
            or audit.get("planned") != 36 or audit.get("terminal") != 36
            or audit.get("known_invalid_raw") != 0):
        raise ValueError(f"{model} supplemental audit not passed")
    if (aggregate.get("status") != "complete" or aggregate.get("method") != "gpu_vector"
            or aggregate.get("control_hz") != 30 or aggregate.get("selection_method") != "supplemental_valid_toss"
            or aggregate.get("base_selection_sha256") != selection["base_selection_sha256"]):
        raise ValueError(f"{model} supplemental aggregate incomplete or wrong method")
    identity = [model_identity(record) for record in (aggregate, audit, launch)]
    if any(item != identity[0] for item in identity[1:]) or identity[0][0] != model:
        raise ValueError(f"{model} supplemental model identity mismatch")
    if (not isinstance(launch.get("started_at"), str) or not launch["started_at"]
            or not isinstance(launch.get("finished_at"), str) or not launch["finished_at"]):
        raise ValueError(f"{model} supplemental launch dates missing")
    if (any(record.get("selection_digest") != selection["selection_digest"]
            for record in (aggregate, audit, launch))
            or any(record.get("selection_sha256") != selection_sha for record in (aggregate, audit, launch))
            or audit.get("aggregate_sha256") != _sha(aggregate_path)
            or audit.get("launch_sha256") != _sha(launch_path)
            or audit.get("run_root") != launch.get("run_root")
            or launch.get("source_hashes") != launch.get("source_hashes_after")
            or not launch.get("source_hashes")):
        raise ValueError(f"{model} supplemental selection/launch/audit provenance mismatch")
    root = Path(audit["run_root"])
    if (not root.is_absolute() or root.resolve() != aggregate_path.resolve().parent
            or root.resolve() != audit_path.resolve().parent
            or root.resolve() != launch_path.resolve().parent
            or audit.get("result_evidence_sha256") != result_evidence_sha256(selection, root, 8)):
        raise ValueError(f"{model} supplemental result evidence changed after audit")
    observed = {}
    for index, row in enumerate(selection["episodes"]):
        result_path = (root / f"shard{index % 8}" / row["split"] / row["task_id"]
                       / row["episode_id"] / "result.json")
        result = _read(result_path)
        if (any(result.get(field) != row.get(field) for field in
                ("split", "task_id", "episode_id", "source_id", "raw_h5", "selection_digest"))
                or model_identity(result) != identity[0]
                or result.get("method") != "gpu_vector"
                or result.get("status") not in ("success", "failure", "unsupported")):
            raise ValueError(f"{model} supplemental result/error/model mismatch: {result_path}")
        observed[row["source_id"]] = result["status"]
    if len(observed) != 36:
        raise ValueError(f"{model} supplemental duplicate source ID")
    for split, quota in (("train", 29), ("val", 7)):
        counts = _counts(aggregate["counts_by_split"][split])
        task = _counts(aggregate["counts_by_task"][split]["toss_in_bin"])
        measured = Counter(observed[row["source_id"]] for row in selection["episodes"] if row["split"] == split)
        if (counts["planned"] != quota or counts["pending"] or counts["error"]
                or any(counts[key] != task[key] for key in _FIELDS)
                or any(counts[key] != audit["counts_by_split"][split][key]
                       or counts[key] != audit["counts_by_task"][split]["toss_in_bin"][key]
                       for key in _FIELDS)
                or any(counts[key] != measured[key] for key in _STATUSES)):
            raise ValueError(f"{model} supplemental audit/result counts mismatch")
    if any(audit.get("status_counts", {}).get(status, 0) !=
           sum(1 for actual in observed.values() if actual == status)
           for status in _STATUSES):
        raise ValueError(f"{model} supplemental audit terminal counts mismatch")
    return aggregate, audit, launch, observed


def compare(base_selection: Path, supplement_selection: Path,
            baseline: dict, mask10: dict) -> dict:
    """Build a comparison only from two distinct, fully audited physical runs per model."""
    base, selection = _read(base_selection), _read(supplement_selection)
    base_sha, selection_sha = _sha(base_selection), _sha(supplement_selection)
    if base_sha != _FROZEN_BASE_SELECTION_SHA256:
        raise ValueError("Historical base selection SHA does not match the frozen 700")
    if (selection.get("selection_method") != "supplemental_valid_toss"
            or selection.get("planned") != 36
            or selection.get("base_selection_sha256") != base_sha
            or Path(selection.get("base_selection_path", "")).resolve() != Path(base_selection).resolve()
            or Counter((row.get("split"), row.get("task_id")) for row in selection.get("episodes", []))
            != Counter({("train", "toss_in_bin"): 29, ("val", "toss_in_bin"): 7})
            or len({row["source_id"] for row in selection["episodes"]}) != 36
            or set(row["source_id"] for row in selection["episodes"]) &
               set(row["source_id"] for row in base["episodes"])
            or selection != build_supplemental_selection(Path(base_selection))):
        raise ValueError("Supplemental selection/base SHA, quota or source identity mismatch")
    evidence = {}
    for model, paths in (("baseline", baseline), ("mask10", mask10)):
        old_final, old_audit, old_aggregate = _old_evidence(paths, model, base, base_sha)
        new_aggregate, new_audit, launch, statuses = _new_evidence(paths, model, selection, selection_sha)
        if old_audit["run_root"] == new_audit["run_root"]:
            raise ValueError("Historical and supplemental results cannot share a run")
        evidence[model] = dict(old_final=old_final, old_audit=old_audit,
                               old_aggregate=old_aggregate, new_aggregate=new_aggregate,
                               new_audit=new_audit, launch=launch, statuses=statuses,
                               input_sha256={name: _sha(Path(path)) for name, path in paths.items()})
    if evidence["baseline"]["new_audit"]["run_root"] == evidence["mask10"]["new_audit"]["run_root"]:
        raise ValueError("Both model supplements must have independent run roots")
    transitions = {split: {} for split in ("train", "val")}
    paired_sources = []
    for row in selection["episodes"]:
        split, source = row["split"], row["source_id"]
        pair = (evidence["baseline"]["statuses"][source], evidence["mask10"]["statuses"][source])
        key = " → ".join(pair)
        transitions[split][key] = transitions[split].get(key, 0) + 1
        paired_sources.append({"split": split, "source_id": source, "baseline_status": pair[0],
                               "mask10_status": pair[1]})
    by_model = {}
    for model, data in evidence.items():
        old, new = data["old_aggregate"], data["new_aggregate"]
        task_splits = {}
        for split, quota, bad in (("train", 500, 29), ("val", 200, 7)):
            old_split = _counts(old["counts_by_split"][split])
            new_split = _counts(new["counts_by_split"][split])
            raw = _sum_counts(old_split, new_split)
            valid = _counts({**raw, "planned": raw["planned"] - bad, "error": 0})
            if valid["planned"] != quota or raw["planned"] != quota + bad:
                raise ValueError("Supplemental denominators do not reconcile")
            by_task = {}
            for task in FIVE_TASKS:
                historical = _counts(old["counts_by_task"][split][task])
                supplement = (_counts(new["counts_by_task"][split][task])
                              if task == "toss_in_bin" else None)
                combined = _sum_counts(historical, supplement) if supplement else historical
                combined_valid = (_counts({**combined, "planned": combined["planned"] - bad,
                                           "error": 0}) if supplement else combined)
                by_task[task] = {"historical": historical, "supplement": supplement,
                                 "combined_raw": combined, "combined_valid_h5": combined_valid}
            task_splits[split] = {
                "historical_700": old_split, "supplement_36": new_split,
                "raw_736_including_original_h5_errors": raw,
                "valid_h5_700_excluding_original_h5_errors": valid, "by_task": by_task}
        totals = {name: _sum_counts(*(task_splits[split][name] for split in ("train", "val")))
                  for name in ("historical_700", "supplement_36", "raw_736_including_original_h5_errors",
                               "valid_h5_700_excluding_original_h5_errors")}
        by_model[model] = {"by_split": task_splits, "totals": totals,
                           "historical_run_root": data["old_audit"]["run_root"],
                           "historical_report_generated_at": data["old_final"]["generated_at"],
                           "supplement_run_root": data["new_audit"]["run_root"],
                           "supplement_started_at": data["launch"]["started_at"],
                           "supplement_finished_at": data["launch"]["finished_at"],
                           "supplement_launch_sha256": data["new_audit"]["launch_sha256"],
                           "input_sha256": data["input_sha256"]}
    return {"status": "final", "selection_sha256": selection_sha,
            "base_selection_sha256": base_sha,
            "supplement_source_ids": {split: [row["source_id"] for row in selection["episodes"]
                                             if row["split"] == split] for split in ("train", "val")},
            "paired_sources": paired_sources, "paired_transitions": transitions, "models": by_model,
            "limitations": ["原700与增补36为两次独立物理运行；raw736仅为组合观察，并非同一次736运行。",
                            "原始H5无效的36条保留在raw计划分母，剔除后才称700条原始H5合格来源。",
                            "两模型的可评分集合可能不同；观察到的成功差异不能单独归因于历史掩码。"]}


def render_chinese(report: dict) -> str:
    lines = ["# toss_in_bin 同源补样对照（独立运行）", "",
             *(f"- {note}" for note in report["limitations"]), "",
             f"- 原700选样SHA256：`{report['base_selection_sha256']}`",
             f"- 两模型共同增补36选样SHA256：`{report['selection_sha256']}`", ""]
    for model, data in report["models"].items():
        lines += [f"## {model}", "",
                  f"- 历史运行：`{data['historical_run_root']}`；历史报告时间：{data['historical_report_generated_at']}",
                  f"- 增补运行：`{data['supplement_run_root']}`；"
                  f"开始：{data['supplement_started_at']}，结束：{data['supplement_finished_at']}；"
                  f"启动证据SHA256：`{data['supplement_launch_sha256']}`",
                  "", "| 来源口径 | split | planned | success | failure | unsupported | error | scored | success/planned | success/scored |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for label in ("historical_700", "supplement_36", "raw_736_including_original_h5_errors",
                      "valid_h5_700_excluding_original_h5_errors"):
            for split, item in (("train", data["by_split"]["train"][label]),
                                ("val", data["by_split"]["val"][label]),
                                ("合计", data["totals"][label])):
                lines.append("| " + " | ".join([label, split, *(str(item[key]) for key in
                    ("planned", "success", "failure", "unsupported", "error", "scored")),
                    *(f"{item[key]:.2%}" if item[key] is not None else "N/A" for key in
                      ("success_per_planned", "success_per_scored"))]) + " |")
        lines += ["", "### 各任务 × split", "", "| split | 任务 | 历史成功/计划/可评分 | 增补成功/计划/可评分 | raw合计成功/计划/可评分 | 合格H5成功/计划/可评分 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for split in ("train", "val"):
            for task, values in data["by_split"][split]["by_task"].items():
                def show(item):
                    return f"{item['success']}/{item['planned']}/{item['scored']}" if item else "—"
                lines.append(f"| {split} | {task} | {show(values['historical'])} | "
                             f"{show(values['supplement'])} | {show(values['combined_raw'])} | "
                             f"{show(values['combined_valid_h5'])} |")
        lines.append("")
    lines += ["## 同一36来源逐项配对转移", "",
              "| split | baseline → mask10 | 个数 |", "| --- | --- | ---: |"]
    for split, transitions in report["paired_transitions"].items():
        for transition, count in sorted(transitions.items()):
            lines.append(f"| {split} | {transition} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--supplement-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for model in ("baseline", "mask10"):
        for name in ("old-final", "old-audit", "old-aggregate", "new-aggregate", "new-audit", "new-launch"):
            parser.add_argument(f"--{model}-{name}", type=Path, required=True)
    args = parser.parse_args()
    def paths(model: str) -> dict:
        return {key: getattr(args, f"{model}_{key}") for key in
                ("old_final", "old_audit", "old_aggregate", "new_aggregate", "new_audit", "new_launch")}
    report = compare(args.base_selection, args.supplement_selection, paths("baseline"), paths("mask10"))
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "comparison.json", report)
    (args.output / "comparison.md").write_text(render_chinese(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
