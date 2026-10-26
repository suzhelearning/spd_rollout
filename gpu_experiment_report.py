"""生成固定535条CPU/GPU实验的中文Markdown及JSON报告（仅处理已有证据）。

示例：python gpu_experiment_report.py --cpu-report CPU/report.json \
    --gpu-report GPU/report.json --gpu-audit gpu-audit.json \
    --benchmark benchmark.json --output REPORT_DIR
benchmark使用{selection_digest, source_ids, horizon_ticks, hardware, variants:[...]}。
每个variant使用method、gpu_count、batch_size、cache_state、elapsed_s、completed、
episodes_per_min及shards；startup/load/loop/policy/peak显存保留在shards中。
缺测字段在JSON中为null、Markdown中为N/A，不以零代替。此CLI不运行模拟或模型。
GPU报告必须由候选aggregate_evaluation.py从逐条result派生method=gpu_vector和
method_evidence；历史CPU报告缺少method时仅按original_cpu解释，不能作为GPU输入。
GPU benchmark variant仅在cache_verified严格为true时参与加速比计算；旧CPU不要求该字段。
提供GPU报告时，必须同时提供通过的--gpu-audit才能标为final；evaluation_final单独保留评估完成度。
GPU审计核对固定digest及planned/terminal/status_counts，保留原始审计及文件SHA256证据。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SELECTION_DIGEST = "2532b02c0bf533cdafefc7213d8cbb8627f1510607cdb4ac9c5ffa6b3b01a841"
TASKS = ("hang_mug", "pyramid", "spelling", "toss_in_bin", "unstack", "vowel_consonant_sort")
EXPECTED = {"train": (45, 51, 8, 234, 46, 44), "val": (12, 13, 1, 59, 12, 10)}
COUNTS = ("planned", "success", "failure", "unsupported", "error", "pending", "scored")
ORIGINAL_METHODS = {"original_cpu", "cpu_original", "original", "cpu_serial_original"}
NOTES = [
    "固定selection为535条，train=428、val=107；error/unsupported保留在planned分母，不替换源样本。",
    "原checkpoint相机标定为provisional；CPU/GPU均沿用--allow-diagnostic并执行完整720tick。结果属于该固定仿真与相机配置下的诊断性闭环评估，不代表相机标定已验证。",
    "success/scored仅衡量可评分回合；success/planned同时反映整份固定选择，scored/planned为评分覆盖率；全终态不等于全部可评分。",
    "旧CPU实验运行中曾优化实现并恢复运行，其全量耗时混合多个版本，不用于推导最初版本加速比；当前source_hashes也不能证明每个历史回合使用相同代码。",
    "主要速度比较仅使用同source_ids、同GPU数、同720tick且缓存条件一致的原始CPU与GPU配对实测；startup、加载/缓存准备和控制循环分别保留，缺测为N/A。",
    "GPU方法复用常驻stage/相机/渲染产品；每批首帧RTX用8子帧清除上一场景残影，后续每次采集用1子帧。该额外首帧渲染成本已计入GPU配对计时，CPU原方法仍保持原采集路径。",
    "原始挂杯单样本175.1968秒仅为历史补充，不作为配对速度证据。",
    "GPU PhysX动力学及并行数值路径不同于原CPU，不声称逐位等价；CPU与GPU成功率独立展示，不能混合汇总或将差异全部归因于策略。",
    "原CPU的78个toss_in_bin错误已逐个只读核查：H5 schema_version=3且内部episode_id与selection/文件名不同；task_id和recorded_action通过。未改raw、映射或评分规则。",
]


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def provenance(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def checked_counts(value: dict, label: str) -> dict:
    result = {}
    for key in COUNTS:
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError(f"{label}.{key} must be a nonnegative integer")
        result[key] = count
    if result["scored"] != result["success"] + result["failure"]:
        raise ValueError(f"{label}: scored differs from success + failure")
    if sum(result[k] for k in ("success", "failure", "unsupported", "error", "pending")) != result["planned"]:
        raise ValueError(f"{label}: terminal + pending differs from planned")
    result.update(success_per_scored=ratio(result["success"], result["scored"]),
                  success_per_planned=ratio(result["success"], result["planned"]),
                  scored_coverage=ratio(result["scored"], result["planned"]),
                  terminal_coverage=ratio(result["planned"] - result["pending"], result["planned"]))
    return result


def evaluation(path: Path | None, method: str) -> dict | None:
    if path is None:
        return None
    raw = load(path)
    if raw.get("selection_digest") != SELECTION_DIGEST:
        raise ValueError(f"{path}: wrong frozen selection digest")
    if raw.get("method", "original_cpu") != method:
        raise ValueError(f"{path}: expected method {method}")
    if raw.get("horizon_ticks") != 720 or raw.get("control_hz") != 30:
        raise ValueError(f"{path}: expected 720 ticks at 30Hz")
    if raw.get("model") != "baseline" or raw.get("checkpoint_step") != 30000:
        raise ValueError(f"{path}: expected baseline checkpoint 30000")
    by_task, by_split = {}, {}
    for split, planned_counts in EXPECTED.items():
        tasks = raw["counts_by_task"][split]
        if set(tasks) != set(TASKS):
            raise ValueError(f"{path}: {split} must contain the six frozen tasks")
        by_task[split] = {}
        for task, expected in zip(TASKS, planned_counts):
            item = checked_counts(tasks[task], f"{split}/{task}")
            if item["planned"] != expected:
                raise ValueError(f"{path}: wrong fixed denominator for {split}/{task}")
            by_task[split][task] = item
        total = checked_counts(raw["counts_by_split"][split], split)
        if any(total[key] != sum(item[key] for item in by_task[split].values()) for key in COUNTS):
            raise ValueError(f"{path}: split totals do not match task totals")
        by_split[split] = total
    total = checked_counts({key: sum(item[key] for item in by_split.values()) for key in COUNTS}, "all")
    method_evidence = raw.get("method_evidence")
    if method == "gpu_vector":
        terminal_count = total["planned"] - total["pending"]
        if (not isinstance(method_evidence, dict)
                or method_evidence.get("source") != "result_records"
                or method_evidence.get("counts") != {"gpu_vector": terminal_count}
                or terminal_count == 0):
            raise ValueError(f"{path}: GPU method must be derived from every terminal result record")
    final = total["planned"] == 535 and total["pending"] == 0
    if raw.get("status") == "complete" and not final:
        raise ValueError(f"{path}: complete status contradicts pending counts")
    timings = []
    for shard in sorted(path.parent.glob("shard*")):
        if not shard.is_dir():
            continue
        runs = sorted(shard.glob("runs/*/timing.json"))
        paths = runs or ([shard / "timing.json"] if (shard / "timing.json").is_file() else [])
        for timing in paths:
            timings.append({"shard": shard.name, "source": provenance(timing), "values": load(timing)})
    launch_path = path.parent / "launch.json"
    launch = load(launch_path) if launch_path.is_file() else None
    return {"method": method, "final": final, "status": "final" if final else "incomplete",
            "method_evidence": method_evidence,
            "total": total, "counts_by_split": by_split, "counts_by_task": by_task,
            "unscored_reasons": raw.get("unscored_reasons", {}), "workers": raw.get("workers", []),
            "timing_runs": timings, "source_hashes": raw.get("source_hashes", {}),
            "wall_clock_seconds": launch.get("elapsed_s") if launch else None,
            "wall_clock_source": provenance(launch_path) if launch else None,
            "source": provenance(path)}


def gpu_audit(path: Path | None, gpu: dict | None) -> dict | None:
    if path is None:
        return None
    if gpu is None:
        raise ValueError("--gpu-audit requires --gpu-report")
    raw = load(path)
    if raw.get("selection_digest") != SELECTION_DIGEST:
        raise ValueError(f"{path}: wrong GPU audit selection digest")
    if raw.get("status") not in ("passed", "failed"):
        raise ValueError(f"{path}: GPU audit status must be passed or failed")
    total = gpu["total"]
    for key, expected in (("planned", total["planned"]),
                          ("terminal", total["planned"] - total["pending"])):
        if type(raw.get(key)) is not int or raw[key] != expected:
            raise ValueError(f"{path}: GPU audit {key} differs from GPU report")
    statuses = raw.get("status_counts")
    terminal_statuses = ("success", "failure", "unsupported", "error")
    if (not isinstance(statuses, dict) or set(statuses) - set(terminal_statuses)
            or any(type(value) is not int or value < 0 for value in statuses.values())
            or any(statuses.get(key, 0) != total[key] for key in terminal_statuses)):
        raise ValueError(f"{path}: GPU audit status_counts differ from GPU report")
    for key in ("full_720tick_90inference_rollouts", "known_raw_errors"):
        if type(raw.get(key)) is not int or not 0 <= raw[key] <= raw["terminal"]:
            raise ValueError(f"{path}: GPU audit {key} must be a terminal rollout count")
    pids = raw.get("worker_pids")
    if (not isinstance(pids, dict)
            or any(not isinstance(values, list)
                   or any(type(pid) is not int or pid <= 0 for pid in values)
                   for values in pids.values())):
        raise ValueError(f"{path}: GPU audit worker_pids must map shards to Isaac PID lists")
    caches = raw.get("cache_sources")
    if (not isinstance(caches, dict)
            or any(type(count) is not int or count < 0 for count in caches.values())):
        raise ValueError(f"{path}: GPU audit cache_sources must contain nonnegative counts")
    initial_error = raw.get("max_initial_error")
    if (initial_error is not None and (type(initial_error) not in (int, float)
            or not math.isfinite(initial_error) or initial_error < 0)):
        raise ValueError(f"{path}: GPU audit max_initial_error must be finite and nonnegative")
    if not isinstance(raw.get("failures"), list):
        raise ValueError(f"{path}: GPU audit failures must be a list")
    if not isinstance(raw.get("run_root"), str) or not raw["run_root"]:
        raise ValueError(f"{path}: GPU audit run_root is required")
    if Path(raw["run_root"]).resolve() != Path(gpu["source"]["path"]).parent.resolve():
        raise ValueError(f"{path}: GPU audit run_root differs from GPU report")
    return raw


def positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def benchmark(path: Path | None) -> dict:
    if path is None:
        return {"status": "unmeasured", "source": None, "variants": [], "comparisons": []}
    raw = load(path)
    if raw.get("selection_digest") != SELECTION_DIGEST:
        raise ValueError(f"{path}: wrong benchmark selection digest")
    variants = []
    for index, value in enumerate(raw.get("variants", [])):
        item = dict(value)
        item["index"] = index
        for key in ("source_ids", "hardware", "horizon_ticks"):
            item[key] = value.get(key, raw.get(key))
        for key in ("method", "gpu_count", "batch_size", "cache_state", "cache_preparation_s",
                    "elapsed_s", "completed", "episodes_per_min"):
            item.setdefault(key, None)
        item.setdefault("shards", [])
        issues = []
        ids = item["source_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids):
            issues.append("缺少唯一的固定source_ids列表")
        if type(item["gpu_count"]) is not int or item["gpu_count"] <= 0:
            issues.append("缺少GPU数量")
        if not item["hardware"]:
            issues.append("缺少硬件信息")
        if item.get("hardware_verified") is not True:
            issues.append("分片实际GPU身份未验证")
        if item["method"] in ORIGINAL_METHODS and item.get("source_verified") is not True:
            issues.append("原始实现缺少运行前源码校验证据")
        if item["horizon_ticks"] != 720:
            issues.append("缺少720tick完整horizon证据")
        if not item["cache_state"] or str(item["cache_state"]).lower() in ("unknown", "n/a"):
            issues.append("缺少cold/cache条件")
        if item["method"] == "gpu_vector" and item.get("cache_verified") is not True:
            issues.append("GPU缓存来源未验证（要求cache_verified为true）")
        if not positive(item["elapsed_s"]):
            issues.append("缺少实测总耗时")
        if type(item["completed"]) is not int or not isinstance(ids, list) or item["completed"] != len(ids):
            issues.append("未完成全部配对样本")
        measured_rate = (item["completed"] * 60 / item["elapsed_s"]
                         if type(item["completed"]) is int and item["completed"] >= 0 and positive(item["elapsed_s"]) else None)
        supplied_rate = item["episodes_per_min"]
        if supplied_rate is not None and (not positive(supplied_rate) or measured_rate is None
                or not math.isclose(supplied_rate, measured_rate, rel_tol=0.01, abs_tol=0.001)):
            issues.append("episodes_per_min与completed/elapsed_s不一致")
        item["episodes_per_min"] = measured_rate
        item["measurement_issues"] = issues
        item["eligible_for_comparison"] = not issues
        variants.append(item)
    comparisons = []
    for original in variants:
        if original["method"] not in ORIGINAL_METHODS:
            continue
        for gpu in variants:
            if gpu["method"] != "gpu_vector":
                continue
            reasons = original["measurement_issues"] + gpu["measurement_issues"]
            for key in ("gpu_count", "hardware", "horizon_ticks", "cache_state"):
                if original[key] != gpu[key]:
                    reasons.append(f"{key}不一致")
            if sorted(original["source_ids"] or []) != sorted(gpu["source_ids"] or []):
                reasons.append("source_ids不一致")
            comparisons.append({"original_index": original["index"], "gpu_index": gpu["index"],
                                "comparable": not reasons, "reasons": list(dict.fromkeys(reasons)),
                                "speedup": original["elapsed_s"] / gpu["elapsed_s"] if not reasons else None})
    return {"status": "measured" if any(row["comparable"] for row in comparisons) else "incomplete",
            "source": provenance(path), "variants": variants, "comparisons": comparisons,
            "manifest": raw}


def cell(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(value) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows), ""]


def render(report: dict) -> str:
    lines = ["# GPU多环境rollout实验报告", "", f"状态：**{report['status']}**；生成时间：{report['generated_at']}。",
             f"评估全终态完成：{report['evaluation_final']}；GPU审计通过：{report['gpu_audit_passed']}。",
             "", f"固定selection digest：`{SELECTION_DIGEST}`。", ""]
    for key, label in (("cpu", "旧CPU全量实验"), ("gpu", "GPU多环境全量实验")):
        data = report[key]
        lines += [f"## {label}", ""]
        if data is None:
            lines += ["N/A：尚未提供该方法的完整评估报告。", ""]
            continue
        lines += [f"方法：`{data['method']}`；状态：{data['status']}。", ""]
        if data["wall_clock_seconds"] is not None:
            lines += [f"本次全量运行墙钟：{cell(data['wall_clock_seconds'])}秒，包含所有分片的启动、加载、控制和退出；分片并行耗时未相加。", ""]
        rows = []
        for split, item in [("总计", data["total"]), *data["counts_by_split"].items()]:
            rows.append([split, *(item[k] for k in COUNTS), percent(item["success_per_scored"]),
                         percent(item["success_per_planned"]), percent(item["scored_coverage"])])
        headers = ["划分", *COUNTS, "success/scored", "success/planned", "scored/planned"]
        lines += table(headers, rows)
        rows = []
        for split, tasks in data["counts_by_task"].items():
            for task, item in tasks.items():
                rows.append([f"{split}/{task}", *(item[k] for k in COUNTS), percent(item["success_per_scored"]),
                             percent(item["success_per_planned"]), percent(item["scored_coverage"])])
        lines += table(headers, rows)
        reasons = [[split, task, reason, count] for split, tasks in data["unscored_reasons"].items()
                   for task, values in tasks.items() for reason, count in values.items()]
        lines += ["未评分原因（error和unsupported均保留）：", ""]
        lines += table(["划分", "任务", "原因", "数量"], reasons)
        if data["timing_runs"]:
            lines += ["分片运行遥测（resume多次调用分别列出；不同分片耗时不可直接相加为墙钟耗时）：", ""]
            timing_keys = ("startup_seconds", "policy_load_seconds", "load_seconds", "loop_seconds", "policy_seconds",
                           "scoring_seconds", "total_seconds", "peak_vram_mib")
            lines += table(["分片/运行", *timing_keys], [[f"{r['shard']}/{Path(r['source']['path']).parent.name}",
                            *(r["values"].get(k) for k in timing_keys)] for r in data["timing_runs"]])
        if key == "gpu":
            lines += ["### GPU执行审计", ""]
            audit = report["gpu_audit"]
            if audit is None:
                lines += ["审计状态：missing；未提供--gpu-audit，整体报告不能标为final。评估全终态完成度单独保留。", ""]
            else:
                lines += [f"审计状态：**{audit['status']}**；终态/planned：{audit['terminal']}/{audit['planned']}；"
                          f"失败项：{len(audit['failures'])}。", "",
                          f"完整720tick且90次推理：{audit['full_720tick_90inference_rollouts']}条；"
                          f"已知raw数据错误：{audit['known_raw_errors']}条（保留在planned分母）。",
                          f"终态分布：{cell(audit['status_counts'])}。",
                          f"缓存来源：{cell(audit['cache_sources'])}；初始状态最大误差："
                          f"{format(audit['max_initial_error'], '.6g') if audit.get('max_initial_error') is not None else 'N/A'}。", ""]
                worker_pids = audit["worker_pids"]
                actual_pid_count = len({pid for pids in worker_pids.values() for pid in pids})
                lines += [f"实际Isaac worker PID总数（去重）：{actual_pid_count}；分片数：{len(worker_pids)}。", ""]
                lines += table(["分片", "实际Isaac PID数量（去重）", "实际Isaac PIDs"],
                               [[shard, len(set(pids)), pids] for shard, pids in sorted(worker_pids.items())])
                lines += [f"审计失败项：{cell(audit['failures'])}。",
                          f"运行目录：{cell(audit['run_root'])}。",
                          f"审计来源及SHA256：{cell(report['inputs']['gpu_audit'])}。", ""]
    lines += ["## 配对速度实验", "", "吞吐量=完整回合数×60/实测墙钟秒；阶段耗时单位秒，显存单位MiB。cold和cached分开比较，缓存准备成本不默认为零。", ""]
    bench = report["benchmark"]
    if not bench["variants"]:
        lines += ["N/A：尚无配对实测；没有生成加速倍数。", ""]
    else:
        lines += ["主比较使用cold_scene_assets，纳入全部场景资产准备成本；warm_scene_assets作为独立缓存重跑展示，不与原始cold直接计算主加速比。场景资产冷加载不代表shader缓存冷启动，shader条件以以下实测设置为准。", ""]
        lines += table(["设置来源", "settings"], [["benchmark", bench["manifest"].get("settings")],
                       *[[f"variant {item['index']}", item.get("settings", bench["manifest"].get("settings"))]
                         for item in bench["variants"]]])
        keys = ("index", "method", "gpu_count", "batch_size", "cache_state", "cache_preparation_s", "completed", "elapsed_s", "episodes_per_min", "hardware")
        lines += table(list(keys), [[item.get(k) for k in keys] for item in bench["variants"]])
        phase_keys = ("gpu_id", "elapsed_s", "startup_s", "load_s", "step_roundtrip_s", "loop_s", "policy_s", "peak_gpu_memory_mib", "gpu_utilization")
        lines += table(["variant", *phase_keys], [[item["index"], *(shard.get(k) for k in phase_keys)]
                       for item in bench["variants"] for shard in item["shards"]])
        lines += ["CPU step_roundtrip_s计入物理、渲染与进程通信；GPU loop_s还包含策略观测与推理，两个阶段字段的范围不同。policy_s为策略观测与推理的合计。", ""]
        preparation = bench["manifest"].get("first_use_preparation", [])
        if preparation:
            lines += ["首次渲染管线准备另行实测，未混入已缓存shader条件下的配对计时：", ""]
            lines += table(["GPU", "首次准备秒数", "用途"],
                           [[item["gpu_id"], item["elapsed_s"], item["purpose"]] for item in preparation])
        lines += table(["原始variant", "GPU variant", "加速倍数", "限制"],
                       [[c["original_index"], c["gpu_index"], c["speedup"], "；".join(c["reasons"]) or "同条件可比较"]
                        for c in bench["comparisons"]])
        for item in bench["variants"]:
            if item["measurement_issues"]:
                lines += [f"- variant {item['index']}：{'；'.join(item['measurement_issues'])}。"]
        lines.append("")
    lines += ["## 解释与限制", "", *(f"- {note}" for note in NOTES), "", "## 输入证据", ""]
    for name, evidence in report["inputs"].items():
        lines.append(f"- {name}：{cell(evidence)}")
    lines += ["", "完整原始benchmark清单、逐方法汇总、运行遥测及审计证据保存在同目录report.json。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path)
    parser.add_argument("--gpu-audit", type=Path, help="GPU执行审计JSON；GPU整体final要求审计通过")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="输出目录，包含report.md和report.json")
    args = parser.parse_args()
    cpu = evaluation(args.cpu_report, "original_cpu")
    gpu = evaluation(args.gpu_report, "gpu_vector")
    gpu_execution_audit = gpu_audit(args.gpu_audit, gpu)
    gpu_audit_passed = bool(gpu_execution_audit is not None
                            and gpu_execution_audit["status"] == "passed"
                            and not gpu_execution_audit["failures"])
    bench = benchmark(args.benchmark)
    audit_path = args.cpu_report.parent / "cpu-audit.json"
    audit = load(audit_path) if audit_path.is_file() else None
    if audit is not None and audit.get("selection_digest") != SELECTION_DIGEST:
        raise ValueError("CPU audit has a different selection digest")
    evaluation_final = bool(cpu["final"] and gpu is not None and gpu["final"])
    report = {"status": "final" if evaluation_final and gpu_audit_passed and bench["status"] == "measured" else "incomplete",
              "evaluation_final": evaluation_final, "gpu_audit_passed": gpu_audit_passed,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "selection_digest": SELECTION_DIGEST, "planned": 535, "cpu": cpu, "gpu": gpu,
              "benchmark": bench, "cpu_audit": audit, "gpu_audit": gpu_execution_audit, "limitations": NOTES,
              "inputs": {"cpu": provenance(args.cpu_report), "gpu": provenance(args.gpu_report),
                         "gpu_audit": provenance(args.gpu_audit),
                         "benchmark": provenance(args.benchmark), "cpu_audit": provenance(audit_path) if audit else None}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "evaluation_final": evaluation_final,
                      "gpu_audit_passed": gpu_audit_passed,
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
