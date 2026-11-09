# CWH v2：完整代码、实验与运行步骤

你的主线保持为：C→W 近似替换 → 回收 C 的原有物理槽位 → 在该槽位加载 H 的 BF16 真副本 → 将 H 的部分工作交给该副本。Custom 和 Native 均为每层 128 个物理槽位；没有额外 Expert 参数槽位，没有修改原模型训练。

这是可审阅、带正确性门禁的研究实现。已完成本地 7 项 CPU 测试、原始结果重算、Python 编译和 Shell 语法检查；本地没有 torch、Triton 或 CUDA，未执行 CUDA 编译、Qwen3 加载、GPU 正确性或大型实验。不能把代码交付当成已经测得加速。

## 1. 包含什么

| 文件 | 用途 |
|---|---|
| `REPORT_CN.md` | 原代码审计、结果修正、相关工作、算法与研究路线 |
| `cwh_v2/policy.py` | 任意 GPU 角色、按层禁用、流式槽位替换与审计 |
| `cwh_v2/routing.py` | Torch 参考路由、单个 Triton 后处理 kernel、Shadow |
| `cwh_v2/model.py` | vLLM 0.13.0 集成、完整 expert_map 检查、加载后张量哈希 |
| `cwh_v2/capture.py` | 独立校准记录、调度 metadata 分相位、训练隐藏状态采样 |
| `tools/quality_pairs.py` | 真实 BF16 SwiGLU 输出上的 C/W 替换误差 |
| `tools/optimize_policy.py` | 逐调用回放、C/W/H/p 联合搜索、验证集准入 |
| `tools/ablation_policies.py` | 相同质量筛选下的频率优先、误差优先解耦基线 |
| `tools/run_suite.py` | 数据准备、校准、10 组诊断、8 组正式实验、Trace、Profiler |
| `tools/gpu_check.py` | Torch/Triton 精确 ID 对照、vLLM 重复 ID 和副本计算检查 |
| `tools/check_artifacts.py` | 跨 GPU 权重哈希一致、四个 Worker 路由一致检查 |
| `tools/summarize_suite.py` | 以三次独立启动为单位统计，输出配对比值区间和质量差异 |
| `tools/recover_v1.py` | 纯 CPU 重算旧实验，不再重跑原 24 轮 |
| `ORIGINAL_RESULTS_RECOMPUTED.json` | 已从真实归档重新算出的旧实验统计 |
| `legacy/`、`cwh_v1_reference/` | 原策略和源码副本；v1 对照只改架构类名以便并存 |

不会安装覆盖你原来的 `cwh_vllm`，不会再次运行旧的 `install_patch.py`。新插件使用独立包名和架构名，通过本包的 dist-info entry point 注册；不需要联网 pip install。

## 2. 环境与路径

沿用归档中真实配置：`vllm/vllm-openai:v0.13.0`，其中记录为 **torch 2.9.0+cu129 / CUDA 12.9**。不要根据其它项目的 torch 2.13 环境升级这个容器。

| 主机路径 | 容器路径 |
|---|---|
| `$SCRATCH/cwh_qwen3_vllm` | `/work/cwh_qwen3_vllm` |
| `$SCRATCH/qwen3_real/models/Qwen3-30B-A3B` | `/work/qwen3_real/models/Qwen3-30B-A3B` |
| `$SCRATCH/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` | `/work/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` |
| `$SCRATCH/openbookqa/validation.jsonl` | `/work/openbookqa/validation.jsonl` |

要求 4×A100-80GB、TP4/EP4、linear placement、BF16、Ag/Rs 后端、enforce-eager。主对照统一关闭 prefix caching 和 chunked prefill。因此新旧运行不能直接跨配置作因果比较；新的 Native / v1 arm 会在完全相同的新配置下重新测量。

## 3. 上传与解压

把 ZIP 放在 Mac 主目录后：

```bash
scp ~/cwh_v2_bundle.zip xinzi@perlmutter.nersc.gov:~/
```

登录 Perlmutter：

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
unzip ~/cwh_v2_bundle.zip
ls cwh_v2_bundle
```

若同名目录已经存在，先另存它再解压，不要将两次代码版本混合。

## 4. 直接提交完整校准

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
sbatch cwh_v2_bundle/nersc.sbatch calibrate cwhv2_a
```

`nersc.sbatch` 沿用了你原脚本的账户 `m4431_g`、regular、`gpu&hbm80g` 和 12 小时。账户若变更，只改 `#SBATCH -A`；输入文件路径若不同，改 `tools/run_suite.py` 的默认路径或直接传参。

这不是小模型试跑。它会直接使用完整 Qwen3-30B-A3B，并自动：

1. 冻结四组会话级不相交 ShareGPT 数据：Train 800、Validation 800、Diagnostic 1600、Test 1600，合计 4800 条不同请求。
2. 每组均包含 Short 与 Long；Train/Validation 各 400+400，Diagnostic/Test 各 800+800。每次生成固定 256 Token。
3. 使用原生权重与原生 Expert IDs 收集 1600 条真实生成请求的校准轨迹。独立记录每个 rank 的文件，避免日志交错；四个 rank 的抽样路由指纹必须一致才能去重 EP0。
4. 从调度 query 长度而非 Batch 大小识别 Decode。该识别依赖已关闭 chunked prefill、Prompt 至少 96 Token；不支持随意改成其它配置。
5. 停止服务后，在 GPU0 上按层读原 safetensors，评估所有 128 个 W 的真实输出；每个 C/目标 GPU 保留误差最小的两个 W。临时输出缓存约 2 GiB（最多 2048 个采样隐藏状态、hidden=2048），不会留在推理模型中。
6. 联合搜索 C/W/H/p，输出 `policies/selected.json`；另生成 `frequency.json`、`quality.json` 两个对照策略。

选择默认门槛（事先固定，不用 Test 调参）：C 的 Decode 频率 ≤ 平均频率的 0.75 倍；C→W 的 Router 加权全层 RMS 重构误差 ≤ 10%；C 的 Router 权重占比 ≤ 1%；训练/验证的 Decode 最大负载代理均值改善均 ≥ 2%；Replica 在相同 C/W 上另有 ≥ 0.5% 的增量改善；Prefill 平均代理回退 ≤ 2%；Decode 调用级 p95 回退 ≤ 10%。这些是工程筛选阈值，不是模型质量或真实延迟的数学保证。

如果输出 0 个启用层，程序会保留完整 Native 层，不能据此宣称 CWH 加速；也不要看完 Test 后悄悄降低阈值。那表示这批校准数据与质量约束下没有找到可靠候选。

查看结果：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['SCRATCH'])/'cwh_qwen3_vllm/cwh_v2_bundle/policies/selected.json'
d=json.loads(p.read_text())
print('Active layers:', sum(x['enabled'] for x in d['layers']), '/ 48')
for x in d['layers']:
    if x['enabled']:
        print(x['layer'], x['cold'], x['warm'], x['hot'], x['replica_fraction'], x['validation_proxy'])
PY
```

## 5. 大型诊断：固定旧 C/W/H，定位实现与分流问题

校准完成后：

```bash
sbatch --array=0-2%1 cwh_v2_bundle/nersc.sbatch diagnostic cwhv2_a
```

每个 array task 独占 4 张 GPU，对所有 arm 各做一次全模型独立启动；3 个 task 对应 3 次重复。`%1` 表示此 array 最多同时运行一个 task，不需要同时占用 12 张 GPU。

| Arm | 作用 |
|---|---|
| Native | 新实验配置的原生 EP 对照 |
| Legacy v1 | 归档中原实现，同配置重测 |
| CW Torch | 去掉 Replica 的匹配对照 |
| Legacy Torch | 保留旧 Logit 分流，整理公共热路径 |
| Legacy Shadow | 相同 Logit Mask 和 where 计算，但不改变 H 的派发 |
| CW Fused | 只有一次融合 C→W 后处理 |
| Legacy Fused | 与 Legacy Torch 相同路由语义，融合执行 |
| p25 Fused | 原布局、可控期望 25% 分流 |
| p50 Fused | 原布局、可控期望 50% 分流 |
| p100 Fused | 原布局、100% 分流 |

0% 对应 CW Fused。25%/50% 是可复现 hash 阈值的期望比例，不能保证小 Batch 恰好转移四分之一/一半 H；正式 Trace 会报告实际比例。

体量：**10 arm × 2 长度 × 800 请求 × 3 独立启动 = 48,000 性能请求，12,288,000 输出 Token**。另每 arm 每重复都跑原 OpenBookQA 500 题。所有 arm 使用相同冻结数据与请求顺序种子，arm 和长度顺序跨重复轮换。

最有解释力的比较：

- Legacy Shadow / CW Torch：额外逻辑对端到端时间的影响；这是匹配对照，不是把总耗时严格分解成可相加分项。
- Legacy Fused / Legacy Torch：在相同派发语义下检验融合收益。
- p25/p50/p100 / CW Fused：控制分流比例，不混入新的 C/W/H 布局。
- v1 / Legacy Torch：公共热路径整理收益，避免把这部分都归因于 Triton。

## 6. 大型正式实验：验证联合选择的算法价值

```bash
sbatch --array=0-2%1 cwh_v2_bundle/nersc.sbatch formal cwhv2_a
```

| Arm | 策略 | 物理槽位/层 |
|---|---|---:|
| Native | 原生 Expert 选择 | 128 |
| EPLB Default | 4 冗余 Expert，window=1000，interval=3000，sync | 132 |
| Fixed Legacy Fused | 原 C/W/H + 原分流规则 + 融合 | 128 |
| Frequency CWH | 相同质量筛选后先选低频 C/W，再独立选高频 H，p=.5 | 128 |
| Quality CWH | 先选最低替换误差 C/W，再独立选高频 H，p=.5 | 128 |
| Selective CW | 联合策略的同一槽位布局，Replica 不接单 | 128 |
| Selective Full | 联合策略的 C/W/H/p + 单 kernel 重定向 | 128 |
| Selective Full Torch | 完全相同策略，用 Torch 执行 | 128 |

Frequency/Quality CWH 是本包自己的解耦对照，不能写成已经复现 CRAFT、SERE 或 R&Q。它们采用相同的候选质量筛选规则；层的启用数量可能不同，需要和质量、启用层数一起报告。

体量：**8 arm × 2 长度 × 800 请求 × 3 独立启动 = 38,400 性能请求，9,830,400 输出 Token**。新 Test 与 Diagnostic、Train、Validation 都不共享会话。所有 formal 策略均由校准生成，不能根据 Diagnostic/Test 结果再次挑选最佳正式候选却仍把原 Test 当未见数据。

这两个性能阶段合计 **86,400 请求、22,118,400 输出 Token**，约为上一轮正式性能体量的 4.5 倍。按旧实验约 300–335 tok/s 粗算，纯性能测量约 18–21 个节点小时（4 GPU/节点），另外还要加启动、质量、校准与追踪。12 小时是每个 array task 的时间，不是整套研究的耗时保证。

## 7. 汇总

两个 array 各自全部完成后，在主机目录运行（这两个汇总仅用标准库）：

```bash
python3 cwh_v2_bundle/tools/summarize_suite.py \
  --root runs/v2_cwhv2_a/diagnostic \
  --out runs/v2_cwhv2_a/diagnostic_summary

python3 cwh_v2_bundle/tools/summarize_suite.py \
  --root runs/v2_cwhv2_a/formal \
  --out runs/v2_cwhv2_a/formal_summary
```

输出 `comparison.csv`、`paired_quality.json`、`interpretation.txt`。三次启动是性能统计单位，800 个请求不能冒充 800 次独立实验。三次测量的配对 log-ratio t 区间只有 df=2，是近似且可能很宽的区间。

同一份 500 题跨重复仍是同一份 500 题，不能说成 1500 道独立题。选项 logprob 差异可以定位行为变化，但不等于完整 logits 或中间激活等价性检查。原来单一任务 Accuracy 的改善仍需要独立任务验证。

## 8. 真实新策略 Trace 与 GPU Profiler

```bash
sbatch cwh_v2_bundle/nersc.sbatch verify cwhv2_a
sbatch cwh_v2_bundle/nersc.sbatch profile cwhv2_a
```

`verify` 用 CW 和 Full 各 800 条 Validation 请求，确认新的 C→W/H→Replica 分流、重复 ID、真实前后负载分布与实际分流比例。它用新的真实运行验证 Native 校准的反事实预测；不是重新运行旧的 800 请求。**带 Capture 的吞吐量绝不能混入性能表。**

`profile` 比较 Native / Selective CW / Selective Full，使用独立 64 请求×32 Token 的固定剖析窗口；这是对正式大型实验的机制采样，不是先跑小性能实验。通过 vLLM 的 `/start_profile`、`/stop_profile` 导出各 Worker 的 torch profiler 时间线。文件可能较大。

```bash
python3 cwh_v2_bundle/tools/profile_summary.py \
  runs/v2_cwhv2_a/profile \
  --out runs/v2_cwhv2_a/profile_summary.json
```

脚本按 kernel 名称启发式统计每文件的重定向、通信、MoE GEMM 和其它 GPU 时间。核函数时间之和可能重叠，不能当关键路径或整体 wall time；关键路径结论仍需检查原始 per-GPU 时间线。不要把 CPU `record_function` 时长误报为 GPU 时长。

## 9. 正确性门禁与失败处理

每次正式阶段自动先执行 GPU gate：多 Batch/多比例下 Torch/Triton ID 精确相同、Python oracle 相同、输入 IDs 不被原地污染、Shadow 不改变 H 派发；另用实际 vLLM `fused_experts` 检查重复 ID 的加权贡献和同权重副本的数值接近性。

v2 每次模型加载都对当前实际线性 expert_map 的 128 项逐个检查，按启用层核验原 C 三个张量确实跳过、H 三个张量确实别名加载，导出 source/replica 的 w13/w2 加载缓冲区 SHA256。四个 rank 的 hash 必须对上。它是加载时字节审计，不冒充完整分布式前向的位级等价证明。

任何门禁失败会在大测量前中止，检查相应 `cuda_correctness.log` 或 `server.log`；不要删掉断言强行跑。若服务因 CUDA/Triton API 失败，这属于尚未验证的环境兼容性，需要修复后重新使用新 tag。已通过的 CPU 检查不代表 GPU 检查已经通过。

程序拒绝复用输出目录，避免旧结果冒充新结果。新一轮用 `cwhv2_b` 等新 tag；若改了策略/筛选阈值，应冻结新数据/新 Test 再报告效果。原始文件和之前的实验结果无需覆盖。

特别注意：**只要某层 C 已被 H 覆盖，即使关闭 Replica，该层也必须继续 C→W。** 不能只把路由开关设为 Native；原 C 权重已不在该 GPU 槽位。v2 的 Native fallback 在加载前决定，禁用层保留原 C 权重与原生路由。这版不实现在线权重迁移、在线完整恢复 C 或 CUDA Graph。

## 10. 成功判据

第一层：GPU gate、权重审计和真正请求 Trace 都正确。第二层：Full 优于相同策略的 CW，证明副本有独立收益；联合策略优于频率/质量解耦策略，证明联合选择有价值。第三层：在固定 128 槽位下稳定优于 Native，同时在多个独立质量任务上满足预先声明的容忍区间。

若只赢过某个 EPLB 配置、只改善 assignments、只恢复到 Native，或启用层为零，都不能写成已证明通用推理加速。报告负结果和适用范围本身也是这一轮实验的有效产出。
