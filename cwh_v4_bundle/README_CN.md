# CWH V4：保持 Cold 槽位回收，改进动态分流及可恢复实验

这是新的研究实现，不是已证明加速的版本。CPU 测试通过不代表 CUDA 编译、模型服务或端到端性能已经通过。交付机器没有 CUDA；本包在分配到的四张 GPU 上先跑强制正确性检查，失败即停止，不会继续烧卡跑长实验。不要把本包的算法代理收益写成实测吞吐。

## 1. 本次修改及研究边界

保留你的完整思路：C→W 近似替代释放 C 的物理槽位，装入 H 的原始 BF16 权重，再把 H assignments 分流到该槽位。每层仍然是 128 个物理槽位，Top-8 和原 router 权重保留，不量化、不训练、不丢弃 assignments。

旧 dynamic 只优化各卡 assignment 数量。新版按当前调用的 expert 直方图，比较五种 H 分流量，成本同时包含活跃专家数量、计算块数量和 token 数。离线用真实 A100 上的本地 fused_experts 测时拟合，再做独立 validation 路由回放否决，最多开启 8 层；无收益的层保留原 C 权重且不安装重定向 callback。

**放置是离线固定的；小调用的分流是在线动态的。** 当前支持每次最多 512 token rows（4096 assignments）的动态决策。更大的调用用训练集挑选的静态分流比例。不能称为“全阶段动态权重迁移”。模型加载后已被 H 覆盖的 C 槽，不能仅靠路由开关恢复 C。在线选择 0% 表示只做 C→W，真正 Native 回退在离线禁用整层时发生。

详细公式和参考论文见 ALGORITHM_CN.md。实验矩阵、组员建议的处理与成功标准见 EXPERIMENTS_CN.md。

## 2. 固定环境

| 项目 | 配置 |
|---|---|
| 主目录 | `/pscratch/sd/x/xinzi/cwh_qwen3_vllm` |
| 新代码 | `$SCRATCH/cwh_qwen3_vllm/cwh_v4_bundle` |
| 容器 | `docker:vllm/vllm-openai:v0.13.0` |
| 模型 | `$SCRATCH/qwen3_real/models/Qwen3-30B-A3B` |
| ShareGPT | `$SCRATCH/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` |
| OpenBookQA | `$SCRATCH/openbookqa/validation.jsonl`，500 题 |
| 硬件 | 单节点 4×A100-80GB，`gpu&hbm80g` |
| 并行 | TP4 + EP4，DP1、PP1，allgather_reducescatter |
| 执行 | BF16、eager、FLASH_ATTN、关闭 prefix cache/chunked prefill |
| KV | 所有组 block size 16、16384 GPU blocks，一致容量 |

宿主 Python 3.6 只做启动时间解析和状态查询。模型和分析均使用容器 Python。不要 `module load pytorch` 后再在宿主运行模型，不要 pip 升级容器，不要加载上次临时 port-fix/sitecustomize 路径。已知容器 torch2.9/CUDA12.9 与宿主模块 torch2.13/CUDA13 不是同一个环境。实际版本写入每次 session 的 environment.json，恢复时核对。

默认数据路径沿用已有实验。缺少文件时 preflight 会在模型加载前报出准确路径，不自动下载模型或换数据。

## 3. VS Code 上传及解压（登录后执行）

在 VS Code Remote SSH 连接 Perlmutter，打开 `/pscratch/sd/x/xinzi/cwh_qwen3_vllm`，把下载的 `cwh_v4_bundle.zip` 拖到这个远程目录。无需放 HOME。旧 runs、旧代码不删除。

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
ls -lh ./cwh_v4_bundle.zip
# 本包使用新目录；如果该目录已有你的 V4 实验代码，先停止，不覆盖运行中的代码。
test ! -e ./cwh_v4_bundle && unzip ./cwh_v4_bundle.zip
python3 ./cwh_v4_bundle/tools/verify_bundle.py
bash ./cwh_v4_bundle/check.sh
```

verify_bundle 校验打包文件 SHA256。check.sh 在容器内跑 CPU 测试、数据/模型/tokenizer 检查，不在登录节点启动模型服务。

## 4. 先选是否复用 V3 校准

默认建议先检查已有 `v3c` 的 train/validation 是否完整。复用只导入校准，不导入旧 formal 性能、QA 或旧 policy；仍重新选择算法策略和测量硬件成本。需要原 calibration `.done_*.json`、attempt capture、workload manifest、study_spec。文件删了或校验失败，不伪造成功。

如果该目录完整，在每次提交/恢复前执行：

```bash
export CWH_IMPORT_V3=/work/cwh_qwen3_vllm/runs/cwh_v3_v3c
```

这里必须是容器 `/work` 路径。若不复用，则在第一次开始新 tag 前执行：

```bash
unset CWH_IMPORT_V3
```

同一个 tag 不能中途从复用切换为不复用；需要切换就用新 tag，例如 v4fresh。代码会检查数据划分 hash 相同，避免旧 train 泄漏到新 test。

## 5. Interactive 四小时

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
salloc -A m4431_g -C 'gpu&hbm80g' -q interactive -N 1 --gpus=4 -t 04:00:00
# 等待 Granted allocation 后执行
bash ./cwh_v4_bundle/launch.sh check v4check 20
bash ./cwh_v4_bundle/launch.sh focus v4a 230
```

`check` 使用单独 tag，只运行 GPU/环境检查，不加载完整模型。focus 自带检查，不强制先单独 check；单独 check 的价值是先看到明确检查结果。启动器按当前 allocation 的真实 EndTime 计算截止时间，因此第一条用了20分钟后第二条不会错误地再假设还有230分钟。

Interactive 不保证关机/断网后存活。想离开电脑，请用下面的 batch。四小时不是全部实验工期，不能保证 focus 跑完。

## 6. 普通十小时，可关机、可断网

退出 interactive allocation（不在运行实验时退出），在登录节点：

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
# 如果第一轮选择复用，恢复前继续设置同一项
export CWH_IMPORT_V3=/work/cwh_qwen3_vllm/runs/cwh_v3_v3c
sbatch -A m4431_g -C 'gpu&hbm80g' -q regular -N 1 --gpus=4 \
  -t 10:00:00 ./cwh_v4_bundle/nersc.sbatch focus v4a
```

若第一轮不复用，则把 export 改为 `unset CWH_IMPORT_V3`。收到 `Submitted batch job ...` 后任务由 Slurm 管理，可以关闭 VS Code、断网或关电脑。八小时只需把 `-t` 改成 `08:00:00`，代码自动按作业结束前五分钟留清理时间。

没有保证十小时一次跑完：focus 共 32 个性能单元，正式生成 6,553,600 tokens；相同 warmup 另约 2,097,152 tokens，300 tok/s 下光这些生成就约 8 小时，另有 6000 QA 请求、校准、多次模型加载。实际 GPU 时间不同，按可恢复任务设计。

## 7. 中断后继续

再次申请 GPU 或再次提交上面**同一个 stage 和 tag** 即可。不要重新解压覆盖，不要改 tag，不要删除 runs，不要并发启动同一 tag。

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
# CWH_IMPORT_V3 与首次相同
sbatch -A m4431_g -C 'gpu&hbm80g' -q regular -N 1 --gpus=4 \
  -t 10:00:00 ./cwh_v4_bundle/nersc.sbatch focus v4a
```

恢复粒度：
- pair quality：每层；硬件 cost：每个 bucket；planner：每层；measured veto：每层。
- 性能：每个 suite/concurrency/repeat/arm/short-or-long 的完整 800 请求。
- QA：每个完整 500 题单元。保存逐题 journal，但中断的 QA attempt 重跑，避免混入重启后不同的 dynamic/EPLB/batching 状态。
- 已完成单元验证 hash 后跳过；未完成单元保留原日志并开新 attempt，不把半轮吞吐与下一轮拼起来。
- 不恢复 CUDA context、KV cache 或半个生成请求。硬断电最多损失当前未完成单元，不承诺保存尚未刷盘的最后日志。
- 同 tag 有文件锁；代码/数据/容器关键版本变化会阻止混合结果，要求新 tag。不要绕过指纹保护。

## 8. 查看状态和汇总

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
squeue -u "$USER"
python3 ./cwh_v4_bundle/tools/status.py ./runs/cwh_v4_v4a
bash ./cwh_v4_bundle/analyze.sh v4a
```

输出到 `runs/cwh_v4_v4a/summary/`：units.csv、ratios.json、pilot_decision.json。部分完成也可查看，未完成不会伪装成完整结果。查看 batch 输出用 `tail -f CWH_v4_JOBID.out`，将 JOBID 替换成提交返回的数字。

main 全部完成后运行完整 paired QA/吞吐统计：

```bash
shifter --image=docker:vllm/vllm-openai:v0.13.0 --volume="$SCRATCH:/work" \
 bash -c 'cd /work/cwh_qwen3_vllm/cwh_v4_bundle; python3 tools/summarize_suite.py --root /work/cwh_qwen3_vllm/runs/cwh_v4_v4a/main/c16 --out /work/cwh_qwen3_vllm/runs/cwh_v4_v4a/summary/main'
```

## 9. 更深入阶段，逐项提交

focus = calibrate → pilot → main。即使 pilot 不好，main 仍原样报告 CWH 候选，避免隐藏负结果；pilot_decision 只建议是否值得采用，不会把 Native 的速度冒充算法结果。

```bash
# 同一 tag、同一 import 设置；每次上一项结束后再提交下一项
sbatch ./cwh_v4_bundle/nersc.sbatch ablation v4a
sbatch ./cwh_v4_bundle/nersc.sbatch regression v4a
sbatch ./cwh_v4_bundle/nersc.sbatch sweep v4a
sbatch ./cwh_v4_bundle/nersc.sbatch profile v4a
```

以上四行不是让你同时运行：按需要逐条提交，避免同 tag 文件锁冲突。`all` 可按序跑所有性能/QA阶段，但不包括 profiler；成本高，建议先看 focus。profile 的输出只能解释瓶颈，不能代替正式吞吐。

若 active_layers=0，这是筛选认为当前设置不值得启用 CWH，必须报告“未找到通过门槛的策略”，不能声称动态算法获胜。
