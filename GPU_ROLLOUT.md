# GPU 常驻多环境 rollout

实现位于 `/root/sunyz/project/spd-rollout-gpu`，与原始 CPU 部署隔离。每张卡运行一个常驻 Isaac Sim 进程和一个共享模型权重的策略进程。批次中的环境同步执行 720 个控制 tick，再加载下一批；各环境的观测历史、动作和随机数流独立。

## 固定实验约束

- `baseline_30000.pt` EMA、相同 DINO 权重、10 diffusion steps。
- 30 Hz 控制，每 tick 统一推进 16 个 PhysX 子步，物理频率 480 Hz。
- `top`、`left_wrist`、`right_wrist` 三路 640×480 RGB，每 8 tick 更新图像和策略动作块，每 tick 更新状态。
- 使用原训练相机配置、JPEG quality=50/subsampling=0、归一化和评分函数。
- 固定 535 条 selection；train=428、val=107；digest 为 `2532b02c0bf533cdafefc7213d8cbb8627f1510607cdb4ac9c5ffa6b3b01a841`。错误与不支持样本保留在分母，不换样本。
- GPU 与 CPU PhysX 的数值路径不同，成功率分别报告。

## 物理与渲染

`gpu_physx_world.py` 在常驻 stage 中维护多环境 articulation/rigid views。环境保留原始坐标，通过成对过滤的 USD CollisionGroups 隔离碰撞。每个控制 tick 只有一次全局 16 子步推进；反馈来自真实 CUDA 张量。切换批次前先停止 PhysX 原生模拟并使张量视图失效，然后替换物理环境；仅使视图失效而删除运行中的 CollisionGroup 会造成未定义行为。多环境 PhysX GPU aggregate-pair 缓冲按环境数配置，避免 SDK 默认容量不足时漏算碰撞。

`gpu_rgb_renderer.py` 复用锁定版本渲染器的几何、材质与相机 authoring。各环境使用独立 RTX scene partition，统一采集 3N 路相机。stage、相机、render product 和 annotator 在批次之间常驻；仅替换各槽位的视觉节点与材质，保留暂存贴图直到 worker 退出。较小的尾批次不销毁多余相机槽位，其分区保持空场景。每批首次采集推进 8 个 RTX 子帧，消除复用 render product 的上一场景残影；后续图像按原每 8 tick 采集、每次 1 子帧。物理几何只作为碰撞体；可视节点由实测 GPU 状态更新，不能用目标动作冒充物理反馈。

GPU 物理只在实际批次 stage 创建后配置。当前 Isaac Sim 6.0.0-rc.59 中，先在启动空 stage 配置 GPU PhysX，再换 stage，会在首次 SyntheticData 图初始化时触发原生崩溃；反复销毁 RTX 产品并切换 stage 也在多批次运行中触发设备丢失。因此批次之间保留 stage 与渲染产品，仅重建物理视图和视觉内容。实际批次选择 CUDA 后关闭 PhysX Fabric，使用 USD 视觉更新；GPU dynamics 与 GPU broadphase 仍启用，并在每次初始化时校验。启动握手中的 `physics_device` 表示目标设备，`physics_initialization=load_batch`；实际启用状态、步长、GPU 缓冲配置与初值误差见每个环境的 `scene_metadata.json`。

## 运行

在部署机器运行，例如 GPU0、8 分片中的第 0 片：

```bash
env CUDA_VISIBLE_DEVICES=0 SPD_ROLLOUT_GPU_ID=0 XDG_RUNTIME_DIR=/run/user/0 \
  /root/sunyz/project/abc/.venv/bin/python \
  /root/sunyz/project/spd-rollout-gpu/gpu_rollout.py \
  --selection /root/sunyz/project/spd-rollout-gpu/evaluation-20pct-selection.json \
  --output /absolute/path/to/new-run/shard0 \
  --shard-count 8 --shard-index 0 --batch-size 8 \
  --dino-frame-batch-size 24 --horizon-ticks 720 \
  --seed 123 --diffusion-steps 10 --allow-diagnostic
```

其他分片分别使用 GPU1–7 与 shard-index 1–7。输出目录首次运行必须不存在。恢复运行使用原命令加 `--resume`；重试已记录的错误再加 `--retry-errors`，全部配置必须与原 session 一致。策略 seed 使用 `123 + 原始分片内样本序号`，批次划分不改变 seed。

该 checkpoint 的相机标定为 provisional，因此完整 720 tick 评估也必须沿用旧 CPU 实验的 `--allow-diagnostic`。这保留原相机配置，不代表标定已经验证。`--source-ids` 只用于诊断子集或配对基准，保留原始分片与 seed；少于 720 tick 的结果不能作为固定完整 horizon 的评估结果。

## 缓存与证据

场景缓存校验原始 H5、scene、机器人、对象资产、相机与 builder 的哈希，记录 `cold_import`、`legacy_initial_usd` 或 `gpu_prepared` 来源。主速度实验用全新缓存目录并设 `SPD_GPU_ALLOW_LEGACY_CACHE=0`，计入真实导入成本；同样样本的缓存重跑另列。shader 缓存状态单独记录。

每回合输出 `result.json`、`scene_metadata.json` 和可评分时的 `final_window.npz`。每次运行输出 `session.json`、`timing.json`、`run_metrics.json`，并在 `runs/` 保存日志和整卡遥测。批次共享时间不能逐回合相加。

全量结束后：

```bash
/root/sunyz/project/abc/.venv/bin/python \
  /root/sunyz/project/spd-rollout-gpu/aggregate_evaluation.py \
  --selection /root/sunyz/project/spd-rollout-gpu/evaluation-20pct-selection.json \
  --run-root /absolute/path/to/new-run --shards 8 --require-complete
```

聚合结果必须无 pending，且拒绝混合 CPU/GPU 方法。全量执行审计还要核对逐源身份和 seed、720 tick/90 次推理、已知无效 H5、每卡常驻 PID/stage/渲染槽位、GPU 缓冲配置及原生日志错误。`gpu_experiment_report.py` 接收 CPU 聚合、GPU 聚合、`--gpu-audit` 与配对 benchmark；仅聚合完整且执行审计通过、速度实测有效时才输出最终报告。同时报告 success/scored、success/planned 和 scored/planned。

## 已执行的集成验证

- 原有测试套件：42 passed。
- 同一 socket worker 连续加载 N=1、N=2，分别推进 120 tick；环境 0 关节和物体轨迹差异均为 0；环境 1 独立动作产生 0.376289 的最大关节状态差异。
- 初始关节、物体位置及四元数误差均小于 9e-8。
- 两路独立策略与批量策略的最大动作差异为 2.384185791015625e-7；缩小批次后重置历史的差异为 0。
- 8 环境重复加载 9 批次，含缩到 1 环境再扩至 8 环境：PID/stage/render products 不变，原生日志无 Error/Fatal；重复相同源的 top 首帧平均像素差最大 3.073/255。
- 原始 8 环境重复加载在第三批曾发生 RTX page fault；6 次 Kit update 和调整销毁顺序都未消除崩溃。常驻 stage 与相机后通过上述 9 批验证，不将以前崩溃后恢复的全量结果标为最终。
- 原始方法与当前GPU方法的同源六任务、双卡、720 tick、冷资产配对计时分别为 684.251 秒与 283.615 秒，2.4126×；GPU缓存资产重跑 276.006 秒，未与冷资产原方法混算。
- 固定535条最终GPU重跑：457条有效源完成720 tick和90次推理；17 success、351 failure、89 unsupported、78条已核实的无效H5 error；墙钟2193.732秒。八个分片各一个常驻Isaac PID/stage，原生日志无Error/Fatal，完整审计为passed。

正式的CPU/GPU分任务成功率、配对速度、原始输入哈希与限制见 `/root/sunyz/project/spd-rollout/outputs/gpu-vector-experiment-20261024/report.md` 和 `report.json`。

## 五任务 100/40 无 RTC 模型对照

五任务与上面的六任务固定 720 tick 实验不是同一口径。首轮五任务实验冻结了同一个 700 来源选择文件（每任务 train 100、validation 40；SHA256 `2327cdd2c4ef2a786e930fbfcaa58f19f6af82f5ff1ae7c8dbac2d204aacbd5f`）和任务上限校准（SHA256 `2036286f4cd4b7ffabbffadb8533b7c33d9d5195a9c665a6ee818b832804da88`），首轮不因可评分性或 H5 错误补样；后续独立补样批次见下节。成功须稳定满足真实物体几何和松手条件 31 个控制帧；否则只在各任务上限 25/50/60/40/105 秒处判 failure 或 unsupported。

`--model baseline` 对应 step 30000 EMA；`--model mask10` 对应历史状态掩码概率为 10% 的 step 30000 EMA，SHA256 `660a0ad5af783e52ed24f5795c2a7f052592ebe2c02f422192ac72f523bbf104`。二者训练数据契约、模型配置、动作归一化和 DINO 权重相同；训练配置中的计划总步数分别是 75000 和 30000，虽然所评 checkpoint 都是第 30000 步，不能将单次成功率差异严格归因于掩码。

推理为无 RTC 串行闭环：每个 30 Hz 控制 tick 反馈状态、更新 KV、提交下一条绝对关节目标并推进 16 个 480 Hz 物理子步；仅在 t=0,8,16… 加入三路新图像并同步生成 `8×54` 动作块。等待新块推理时不推进物理，所以仿真时间中生成块的频率是 3.75 Hz，而非墙钟实时 30 Hz 推理。

运行时始终为每个模型使用独立的空输出根；在冻结 selection.json 与 calibration.json 的根目录调用 `five_task_experiment.py --model mask10 --code-root /root/sunyz/project/spd-rollout-gpu-mask10 --selection <根>/selection.json --evidence-root <根>`，随后对 `<根>/gpu-full` 执行 `five_task_audit.py --model mask10 --selection <根>/selection.json --run-root <根>/gpu-full --shards 8 --assets /root/sunyz/project/spd-rollout/assets`。审计要核对 checkpoint 与启动文件哈希、逐来源真实物理评分、状态/KV/动作与物理频率、推理次数和常驻 GPU worker；`error`、`unsupported` 始终留在 planned 分母而不混作 failure。只有审计状态 `passed` 且报告状态 `final` 才引用成功率。

2026-10-25 的 mask10 独立八卡 700 条正式运行已完成：13 success、559 failure、92 unsupported、36 条冻结原始 H5 error；572 条 scored，train 成功 11/406、validation 成功 2/166，墙钟 6504.875 秒。逐源审计重放 664 条实测评分窗口，状态 `passed` 且 `failures: []`，机读和 Markdown 报告状态均为 `final`。原始证据位于 `/root/sunyz/project/spd-rollout/outputs/five-task-mask10-100x40-early-stop-20261025/`；与同来源 baseline 的配对终态及两种分母对照见该目录下的 `comparison.md`、`comparison.json`。不能把观察到的成功数差异或八卡墙钟差异当作单一训练因素的因果效应。

## toss_in_bin 原始 H5 错误的独立补样批次

历史两模型各有 700 个固定原来源，其中 toss_in_bin train 29、val 7 个 H5 是 v3 旧别名：在未进入物理前报错。本次保留原 700 来源和已审计结果，不改写、不删 error；从同一训练快照和独立排序中，在旧 700 来源之外各选首 29/7 个 H5 头部及场景校验通过的 toss 来源，不按评分、成功率或模型挑样。补样清单 SHA256 `6efc8c31e6e974e6647c9d3ae69e6f3fe54569c727c0eb4dea1e0022e46cce0f`，选择摘要 `62c0009dccabdbc0c3cc549ab2ddfa1746286d339bd08d0ef729c41d0597de48` 还绑定逐个选定 H5 的 SHA256。baseline/mask10 对同一份清单在不同运行根各执行 36 个真实八卡 GPU PhysX 回合，保持原有无 RTC、30/480 Hz、每 8 tick 新动作块、toss 60 秒安全上限和连续 31 帧目标确认。两轮新增回合均 `error=0`、独立逐源审计 `passed` 且 36 个评分窗口重新计算。

新增 36 中 baseline 成功 10（train 8、val 2），mask10 成功 13（train 10、val 3）；旧首轮成功分别 23、13。含旧错误的 raw 组合观察是 736 条，baseline 成功 33/736＝4.48%，mask10 成功 26/736＝3.53%；只剔除旧 36 条原始 H5 错误得到 H5 合格来源 700 条，baseline 33/700＝4.71%，mask10 26/700＝3.71%。可评分分母另列：baseline 33/601＝5.49%，mask10 26/603＝4.31%；unsupported 不得冒充 failure 或静默丢弃。736 指原700与新增36的两个独立运行合并观察，**不是**一次736来源完整仿真，也不允许把新的 H5 合格700 率冒称历史原700率。相机仍采用 provisional 校准；任务安全上限取 train＋validation 成功演示 P95，validation 非完全盲评；两轮差异不能归因于单一掩码因素。

可复算入口为 `supplement_selection.py --base-selection <历史700选择> --output <新独立根>/selection.json`；确认其 SHA 后，分别将相同选择文件放在两个不同模型的实验根，使用 `five_task_experiment.py --model baseline|mask10 --code-root /root/sunyz/project/spd-rollout-gpu-supplement --selection <模型根>/selection.json --evidence-root <模型根>` 并分别执行 `five_task_audit.py --model <对应模型> --selection <模型根>/selection.json --run-root <模型根>/gpu-full --shards 8 --assets /root/sunyz/project/spd-rollout/assets`。原始逐来源结果、两次启动哈希/物理审计及四种分母的完整对照在 `/root/sunyz/project/spd-rollout/outputs/toss-supplement-29x7-verified-20261025/` 的 `baseline/gpu-full/`、`mask10/gpu-full/` 和 `comparison/`。两份用户自有原飞书文档已分别保留首轮表格与附件，在文末增补新证据和有效来源分母。
