#!/usr/bin/env python3
"""Build the audited five-task rollout document without mixing legacy outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from aggregate_evaluation import legacy_baseline_identity, model_identity
from five_task_audit import result_evidence_sha256

from eval_selection import FIVE_TASKS
from rollout_policy import save_json
from rollout_termination import TASK_LIMIT_TICKS

_COUNT_FIELDS = ("planned", "success", "failure", "unsupported", "error", "pending", "scored")
_LIMITATIONS = (
    "checkpoint相机标定为provisional，实验沿用--allow-diagnostic；不代表相机标定已验证。",
    "GPU PhysX与其他仿真器的动力学数值路径不同；本实验只有GPU五任务结果，不能把旧六任务或旧固定时长结果当同条件对照。",
    "30 Hz仅为仿真时间中的状态/KV/绝对动作控制频率；无RTC，同步等待每8步一次的8×54新动作块推理，推理期间物理暂停；仿真时间动作块生成频率为3.75 Hz，不承诺墙钟实时。",
    "成功要求连续1秒实测目标和松手；上限未达标才判failure，原始H5错误与无法验证的目标保留在planned分母而不参与scored分母。",
    "任务安全上限使用train与validation成功演示的P95共同校准；用户明确保留vowel_consonant_sort的105秒上限，validation结果不是完全未触及validation统计量的盲评。",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _counts(item: dict) -> dict:
    result = {name: item[name] for name in _COUNT_FIELDS}
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError("Evaluation counts must be nonnegative integers")
    if (result["scored"] != result["success"] + result["failure"]
            or result["planned"] != sum(result[key] for key in
                                         ("success", "failure", "unsupported", "error", "pending"))):
        raise ValueError("Evaluation counts do not partition the fixed planned sources")
    return {**result, "success_per_scored": _ratio(result["success"], result["scored"]),
            "success_per_planned": _ratio(result["success"], result["planned"]),
            "scored_coverage": _ratio(result["scored"], result["planned"])}


def summarize_evidence(selection: dict, aggregate: dict, audit: dict,
                       calibration: dict, launch: dict) -> dict:
    issues = []
    if (selection.get("selection_method") != "fixed_count_per_task_split"
            or tuple(selection.get("tasks", ())) != FIVE_TASKS
            or selection.get("counts_per_task_split") != {"train": 100, "val": 40}
            or selection.get("by_split") != {"train": 500, "val": 200}
            or selection.get("planned") != 700):
        issues.append("wrong_selection_contract")
    digest = selection.get("selection_digest")
    if (aggregate.get("selection_digest") != digest or audit.get("selection_digest") != digest
            or aggregate.get("method") != "gpu_vector" or aggregate.get("control_hz") != 30
            or aggregate.get("status") != "complete"):
        issues.append("evaluation_provenance_or_completeness_mismatch")
    identity = None
    records = (aggregate, audit, launch)
    legacy = legacy_baseline_identity(launch) and "code_root" not in launch
    try:
        claimed = [model_identity(record, legacy_baseline=legacy) for record in records]
        identity = claimed[0]
        if any(item != identity for item in claimed[1:]):
            identity = None
            issues.append("evaluation_model_identity_mismatch")
    except ValueError:
        issues.append("evaluation_model_identity_mismatch")
    if identity is None:
        identity = ("unknown", None, None)
    by_task, by_split = {}, {}
    for split, quota in (("train", 100), ("val", 40)):
        source = aggregate["counts_by_task"][split]
        if set(source) != set(FIVE_TASKS):
            issues.append(f"{split}_task_set_mismatch")
        by_task[split] = {task: _counts(source[task]) for task in FIVE_TASKS}
        if any(row["planned"] != quota for row in by_task[split].values()):
            issues.append(f"{split}_quota_mismatch")
        by_split[split] = _counts(aggregate["counts_by_split"][split])
        if any(by_split[split][key] != sum(row[key] for row in by_task[split].values())
               for key in _COUNT_FIELDS):
            issues.append(f"{split}_totals_mismatch")
    total = _counts({key: sum(row[key] for row in by_split.values()) for key in _COUNT_FIELDS})
    audited_tasks = audit.get("counts_by_task", {})
    audited_splits = audit.get("counts_by_split", {})
    if any(
        set(audited_tasks.get(split, {})) != set(FIVE_TASKS)
        or any(any(audited_tasks[split][task].get(key) != by_task[split][task][key]
                   for key in _COUNT_FIELDS) for task in FIVE_TASKS)
        or any(audited_splits.get(split, {}).get(key) != by_split[split][key]
               for key in _COUNT_FIELDS)
        for split in ("train", "val")
    ):
        issues.append("audit_task_counts_mismatch")
    if (audit.get("run_root") != launch.get("run_root")
            or audit.get("selection_sha256") != launch.get("selection_sha256")):
        issues.append("audit_run_identity_mismatch")
    if (total["planned"] != 700 or total["pending"] != 0
            or audit.get("status") != "passed" or audit.get("failures")
            or audit.get("planned") != total["planned"]
            or audit.get("terminal") != total["planned"]
            or any(audit.get("status_counts", {}).get(status, 0) != total[status]
                   for status in ("success", "failure", "unsupported", "error"))):
        issues.append("execution_audit_or_status_count_mismatch")
    caps = calibration.get("tasks", {})
    if (calibration.get("source_splits") != ["train", "val"]
            or calibration.get("scope") != "train_and_validation_source_demonstrations"
            or set(caps) != set(FIVE_TASKS)
            or any(caps[task].get("limit_ticks") != limit for task, limit in TASK_LIMIT_TICKS.items())):
        issues.append("task_cap_calibration_mismatch")
    shards = launch.get("shards", [])
    if (not isinstance(shards, list) or len(shards) != 8
            or {entry.get("gpu") for entry in shards} != set(range(8))
            or any(entry.get("exit_code") not in (0, 1) for entry in shards)
            or not isinstance(launch.get("elapsed_s"), (int, float))
            or not math.isfinite(launch["elapsed_s"]) or launch["elapsed_s"] <= 0):
        issues.append("shard_launch_or_wall_time_mismatch")
    return {"status": "final" if not issues else "incomplete", "issues": issues,
            "model": identity[0], "checkpoint_sha256": identity[1],
            "history_mask_prob": identity[2],
            "selection_digest": digest, "planned": 700,
            "counts_by_task": by_task, "counts_by_split": by_split, "total": total,
            "task_limits": caps, "elapsed_s": launch.get("elapsed_s"),
            "limitations": [*_LIMITATIONS, *(
                ["baseline训练计划配置75000步、mask10训练计划配置30000步，虽均取step30000 EMA且数据契约/模型配置相同，效果差异不能单独归因于历史掩码。"]
                if identity[0] == "mask10" else [])]}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _provenance(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _cell(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")

def _percent(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "N/A"


def _rate_row(label: str, item: dict) -> list:
    return [label, *(item[key] for key in
                     ("planned", "success", "failure", "unsupported", "error", "scored")),
            *(_percent(item[key]) for key in
              ("success_per_scored", "success_per_planned", "scored_coverage"))]


def _table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)]


def _dates(selection: dict) -> list[dict]:
    dates = defaultdict(Counter)
    for row in selection["episodes"]:
        match = re.match(r"^episode_(\d{8})_", row["episode_id"])
        dates[match.group(1) if match else "无日期"][row["split"]] += 1
    return [{"date": date, "train": count["train"], "val": count["val"],
             "total": count["train"] + count["val"]}
            for date, count in sorted(dates.items())]


def _terminal_times(selection: dict, run_root: Path, shards: int) -> dict:
    observations = defaultdict(list)
    for index, row in enumerate(selection["episodes"]):
        path = run_root / f"shard{index % shards}" / row["split"] / row["task_id"] / row["episode_id"] / "result.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        if type(result.get("tick")) is int and result["tick"] > 0:
            observations[(row["split"], row["task_id"], result.get("termination_reason", "unclassified"))].append(result["tick"])
    return {f"{split}/{task}/{reason}": {"count": len(ticks),
             "min_tick": min(ticks), "max_tick": max(ticks),
             "median_simulated_seconds": float(median(ticks)) / 30}
            for (split, task, reason), ticks in sorted(observations.items())}


def render(report: dict, aggregate: dict, audit: dict, calibration: dict) -> str:
    model = report["model"]
    probability = report["history_mask_prob"]
    mask_description = f"{probability:.0%}" if probability is not None else "unknown"
    lines = ["# 五任务 GPU 物理闭环实验", "", f"状态：**{report['status']}**；生成于 {report['generated_at']}。",
             "", f"模型：**{model}**（历史掩码概率 {mask_description}）；checkpoint SHA256：`{report['checkpoint_sha256']}`。",
             f"选样摘要：`{report['selection_digest']}`；planned：700（train 500、validation 200）。",
             f"八卡墙钟：{_cell(report['elapsed_s'])} 秒（分片并行耗时未相加）。", "",
             "## 仿真与自动终止", "",
             f"物理 480 Hz，每控制周期 16 子步；状态反馈/KV/绝对关节动作 30 Hz；无RTC，同步等待每8步生成的一个 8×54 新动作块，推理期间物理暂停；仿真时间块生成频率 3.75 Hz，三路相机每 8 tick 更新。模型为 {model} step30000 EMA、10 次 diffusion 采样。", "",
             "连续 31 个真实控制帧（1 秒）经原任务几何评分器确认目标稳定且松手后即成功；到各任务上限仍未达目标才记 failure，证据不足记 unsupported，原始数据或运行错误记 error。", ""]
    lines += _table(["Task", "有效成功演示", "P95 秒", "上限秒", "上限 tick"],
                    [[task, value["valid_success_demos"], value["p95_s"], value["limit_seconds"], value["limit_ticks"]]
                     for task, value in sorted(calibration["tasks"].items())])
    lines += ["", "## 五任务 train / validation 结果", "",
              "成功率同时使用 success/scored 与 success/planned 两种分母；unsupported/error 不计入 scored，但保留在 planned。", ""]
    headers = ["划分 / Task", "planned", "success", "failure", "unsupported", "error", "scored",
               "success/scored", "success/planned", "scored/planned"]
    rows = []
    for split in ("train", "val"):
        for task in FIVE_TASKS:
            item = report["counts_by_task"][split][task]
            rows.append(_rate_row(f"{split}/{task}", item))
        item = report["counts_by_split"][split]
        rows.append(_rate_row(split + " 合计", item))
    rows.append(_rate_row("总计", report["total"]))
    lines += _table(headers, rows)
    lines += ["", "## 来源日期与终止证据", "",
              "日期仅取 episode_id；无日期旧 ID 不臆断采集时间。", ""]
    lines += _table(["日期", "train", "validation", "合计"],
                    [[d["date"], d["train"], d["val"], d["total"]] for d in report["source_dates"]])
    lines += ["", f"逐源审计：**{audit['status']}**；终态 {audit['terminal']}/{audit['planned']}；重放实测评分窗口 {audit['replayed_physical_rollouts']} 条；已知无效原始 H5 {audit['known_invalid_raw']} 条。", ""]
    lines += _table(["划分 / Task / 终止原因", "数量", "最早 tick", "最晚 tick", "模拟时长中位数（秒）"],
                    [[key, value["count"], value["min_tick"], value["max_tick"], value["median_simulated_seconds"]]
                     for key, value in report["terminal_times"].items()])
    lines += ["", "## 未评分原因", ""]
    reason_rows = [[split, task, reason, count]
                   for split, tasks in aggregate.get("unscored_reasons", {}).items()
                   for task, reasons in tasks.items() for reason, count in reasons.items()]
    lines += _table(["划分", "Task", "原因", "数量"], reason_rows)
    lines += ["", "## 输入及运行证据", ""]
    for name, evidence in report["inputs"].items():
        lines.append(f"- {name}: `{evidence['path']}` SHA256 `{evidence['sha256']}`")
    lines += [f"- 代码／checkpoint 启动哈希：`{json.dumps(report['source_hashes'], ensure_ascii=False, sort_keys=True)}`",
              f"- 分片整卡指标：`{json.dumps(report['worker_metrics'], ensure_ascii=False)}`", "",
              "## 限制", "", *(f"- {note}" for note in report["limitations"]), ""]
    if report["issues"]:
        lines += ["## 未完成验收", "", *(f"- {issue}" for issue in report["issues"]), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "aggregate", "audit", "calibration", "launch", "run_root", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("selection", "aggregate", "audit", "calibration", "launch")}
    sources = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    report = summarize_evidence(*(sources[name] for name in inputs))
    launch = sources["launch"]
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["inputs"] = {name: _provenance(path) for name, path in inputs.items()}
    selection_path = args.selection.resolve()
    run_path = args.run_root.resolve()
    if (sources["audit"].get("run_root") != str(run_path)
            or launch.get("run_root") != str(run_path)
            or launch.get("selection_path") != str(selection_path)
            or sources["audit"].get("selection_sha256") != _sha(args.selection)
            or launch.get("selection_sha256") != _sha(args.selection)
            or sources["audit"].get("aggregate_sha256") != _sha(args.aggregate)
            or sources["audit"].get("launch_sha256") != _sha(args.launch)
            or sources["audit"].get("result_evidence_sha256") !=
            result_evidence_sha256(sources["selection"], run_path, 8)):
        report["issues"].append("audit_stale_or_wrong_run")
    report["source_dates"] = _dates(sources["selection"])
    report["terminal_times"] = _terminal_times(sources["selection"], args.run_root, 8)
    report["source_hashes"] = launch.get("source_hashes", {})
    recorded_code_root = launch.get("code_root")
    if isinstance(recorded_code_root, str) and Path(recorded_code_root).is_absolute():
        code_root = Path(recorded_code_root)
    elif legacy_baseline_identity(launch) and "code_root" not in launch:
        code_root = Path("/root/sunyz/project/spd-rollout-gpu")
    else:
        code_root = Path(__file__).resolve().parent
        report["issues"].append("missing_launch_code_root")
    for name, recorded in report["source_hashes"].items():
        path = Path(name) if Path(name).is_absolute() else code_root / name
        if not path.is_file() or _sha(path) != recorded:
            report["issues"].append(f"source_hash_mismatch:{name}")
    if not report["source_hashes"]:
        report["issues"].append("missing_startup_source_hashes")
    report["worker_metrics"] = {}
    for shard in range(8):
        path = args.run_root / f"shard{shard}" / "timing.json"
        if not path.is_file():
            report["issues"].append(f"missing_shard_timing:{shard}")
            continue
        timing = json.loads(path.read_text())
        report["worker_metrics"][str(shard)] = {key: timing.get(key) for key in
            ("total_seconds", "startup_seconds", "load_seconds", "loop_seconds",
             "policy_seconds", "scoring_seconds", "peak_vram_mib", "mean_gpu_utilization_percent")}
    if report["issues"]:
        report["status"] = "incomplete"
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report.json", report)
    (args.output / "report.md").write_text(
        render(report, sources["aggregate"], sources["audit"], sources["calibration"]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": report["issues"],
                      "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)
    if report["status"] != "final":
        raise SystemExit("Five-task experiment has not passed full source and runtime audit")


if __name__ == "__main__":
    main()
