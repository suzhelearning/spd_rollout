# SPD GPU rollout（5090-104）

本仓库保存 5090-104 上的 SPD 闭环 GPU PhysX rollout **代码**，包括五任务 baseline/mask10 对照和 `toss_in_bin` 独立补样。它是已有机器部署的源码快照，不包含训练代码、模型权重、DINO 权重、原始 H5、仿真资产或运行证据；克隆本仓库本身不能直接重跑实验。较早的六任务固定 720 tick 实验与 2026-10-25 的五任务早停实验口径不同，不合并成功率。

## 代码路径

| 路径 | 职责 |
| --- | --- |
| `five_task_selection.py`、`five_task_calibration.py` | 冻结五任务来源及按任务配置的早停上限。 |
| `supplement_selection.py`、`eval_selection.py` | 从原选择以独立排序选取 29 train / 7 validation 条原始 H5 合格的 `toss_in_bin` 来源。 |
| `five_task_experiment.py` | 校验冻结清单和权重哈希，在八张物理 GPU 上启动分片并聚合结果。 |
| `gpu_rollout.py`、`gpu_physx_worker.py`、`gpu_physx_world.py`、`gpu_rgb_renderer.py` | 闭环策略、常驻 Isaac Sim GPU 物理、物体反馈及三路 RGB 相机。 |
| `rollout_policy.py`、`infer_recording.py`、`rollout_termination.py` | 校验并加载模型、构造推理输入、控制早停和任务时限。 |
| `success_blocks.py`、`success_vessels.py` | 从实测物体几何与松手状态判定目标，不以命令目标冒充结果。 |
| `aggregate_evaluation.py`、`five_task_audit.py`、`five_task_report.py` | 汇总终态，逐来源核对启动哈希、PhysX 证据和评分窗口，审计通过后生成正式报告。 |
| `supplement_compare.py` | 核对两模型原 700 与新增 36 的独立证据，分别报告原计划、补样、raw 组合及原始 H5 合格来源口径。 |
| `tests/` | 选择、模型身份、成功判据、终止、审计与补样对照的回归测试。 |
| `GPU_ROLLOUT.md` | 历史六任务 GPU 实现、运行约束与五任务及补样实验详情。 |

## 环境与运行约束

机器上的 `abc/`、`assets/`、`checkpoints/`、`data_collection/`、`dino/`、`runtime/`、`samples/`、`spd_source/` 是指向其他部署位置的软链，**未提交**。部署时需分别提供兼容的 ABC/SPD 训练代码和 Python 环境、Isaac Sim 6.0.0-rc.59、场景与原始 H5、两份 step 30000 EMA checkpoint、DINOv3 权重；`infer_recording.py` 会校验权重 SHA256。原机默认 Python 为 `/root/sunyz/project/abc/.venv/bin/python`，程序运行依赖八张可见 GPU、对应的资产路径和独立空输出根。不要将本仓库中的源码与其他部署版本的资产/权重未经校验混用。

五任务选择文件固定每任务 train 100、validation 40，共 700 个来源；selection SHA256 为 `2327cdd2c4ef2a786e930fbfcaa58f19f6af82f5ff1ae7c8dbac2d204aacbd5f`。历史错误仍在计划分母。两模型共用来源、任务上限及仿真约束：30 Hz 状态反馈与绝对关节目标，480 Hz PhysX（每 tick 16 子步），每 8 tick 为三路相机采图并同步推理一个 `8×54` 动作块；推理期间不推进物理，不承诺墙钟实时控制。成功要求真实物体几何及松手条件连续 31 个控制帧达标；各任务上限为 25/50/60/40/105 秒。

在原机已有资产和冻结清单时，运行入口为：

```bash
PY=/root/sunyz/project/abc/.venv/bin/python
CODE=/root/sunyz/project/spd-rollout-gpu-supplement
# 各模型使用不同且尚不存在输出的实验根；先将已冻结的 selection.json、calibration.json 放在实验根。
$PY "$CODE/five_task_experiment.py" --model baseline --code-root "$CODE" \
  --selection /absolute/new-baseline-run/selection.json --evidence-root /absolute/new-baseline-run
$PY "$CODE/five_task_audit.py" --model baseline \
  --selection /absolute/new-baseline-run/selection.json \
  --run-root /absolute/new-baseline-run/gpu-full --shards 8 \
  --assets /root/sunyz/project/spd-rollout/assets
```

`--model mask10` 在**另一个**实验根执行相同流程。只有 `audit.json` 的 `status=passed` 且正式 `report.json` 的 `status=final` 时才引用成功率；生成报告、增补来源及四种分母的命令与限制见 [GPU_ROLLOUT.md](GPU_ROLLOUT.md)。日常代码回归可在已有部署环境执行：`$PY -m pytest -q tests`。

## 2026-10-25 已核实结果

以下为**原始 700 来源各独立运行一次**，每任务计划 140 条（train 100 + validation 40）。`scored = success + failure`；`unsupported` 与 `error` 保留在 planned 中，不伪装成 failure。

| Task | baseline 成功/计划 | baseline 成功/可评分 | mask10 成功/计划 | mask10 成功/可评分 |
| --- | ---: | ---: | ---: | ---: |
| hang_mug | 0/140 | 0/140 | 0/140 | 0/140 |
| pyramid | 0/140 | 0/140 | 0/140 | 0/140 |
| toss_in_bin | 23/140 | 23/82 | 13/140 | 13/73 |
| unstack | 0/140 | 0/138 | 0/140 | 0/140 |
| vowel_consonant_sort | 0/140 | 0/74 | 0/140 | 0/79 |
| **总计** | **23/700 = 3.29%** | **23/574 = 4.01%** | **13/700 = 1.86%** | **13/572 = 2.27%** |

原 700 各有 36 条冻结来源的原始 H5 错误。针对 `toss_in_bin` 在原 700 之外独立补样 36 条：baseline 成功 10/36、mask10 成功 13/36，两轮均无 H5 错误。原 700 + 增补 36 是 **两次独立运行的组合观察**：保留旧错误为 33/736、26/736；剔除旧 36 条 H5 错误后的合格来源为 33/700、26/700，**不是原始 700 来源的成功率**。四轮审计均 `passed`、`failures=[]`，正式报告为 `final`。

原始结果、逐回合物理证据与核对资料仍在 5090-104，未上传 Git：

- `/root/sunyz/project/spd-rollout/outputs/five-task-100x40-early-stop-verified-20261025/`
- `/root/sunyz/project/spd-rollout/outputs/five-task-mask10-100x40-early-stop-20261025/`
- `/root/sunyz/project/spd-rollout/outputs/toss-supplement-29x7-verified-20261025/comparison/`

相机标定仍为 provisional，任务上限用 train 与 validation 的成功演示共同校准；validation 并非严格盲评。两模型虽均为 step 30000 EMA，但单次成功数差异不能单独归因于历史掩码。
