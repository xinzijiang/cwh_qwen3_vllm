> 审计更新：修复 Prefill 采样周期混叠，补齐行纳入动态回放。首次运行可沿用 v3a；若旧版本已开始实验，请保留旧目录，新版用新 tag（例如 v3b）。

# CWH v3：动态分流 + 可恢复的 Perlmutter 实验

本包更新 v2：在线按当前调用的真实 assignments 决定 H→Replica 数量；保留 C→W 回收槽位的原主线。还修复 v2 静态 kernel factory 的未定义返回变量，并补齐逐请求记录和断点恢复。不要再按 history/ 中旧 README 的不可恢复流程启动。

已通过本地15项 CPU 测试，包括15379组闭式解/整数穷举对照、混合相位回放、日志截断修复，以及模拟中断后跳过已完成两轮测试的工作流测试。没有本地 torch/CUDA，GPU kernel 尚未实际编译执行；每次任务会先运行 `gpu_check.py`，通过后才做大型实验。

算法公式、最优性的严格适用范围和相关方法比较见 `ALGORITHM_CN.md`。内部 Python 包/架构仍名为 cwh_v2，以保持 API 兼容；当前分发版本是0.3.0。

## 上传与解压（一次）

把下载的 `cwh_v3_bundle.zip` 用 Finder 放到 Mac 主目录，避免之前 Downloads 权限问题。在 Mac 终端：

```bash
scp ~/cwh_v3_bundle.zip xinzi@perlmutter.nersc.gov:~/
ssh xinzi@perlmutter.nersc.gov
```

在 Perlmutter 登录节点：

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
unzip ~/cwh_v3_bundle.zip
```

目录是新的 `cwh_v3_bundle`，不覆盖你的原 `cwh_vllm` 或 v2。若同名v3目录已经存在，勿混入不同代码版本；使用独立目录/新研究tag。恢复过程中不要修改代码和数据。

继续使用原 `vllm/vllm-openai:v0.13.0` Shifter 镜像和 ROOT/vendor；不要切换到其它实验的 torch 环境。原归档容器记录为 torch2.9.0+cu129、vLLM0.13.0。

预期文件：

```text
$SCRATCH/qwen3_real/models/Qwen3-30B-A3B/config.json
$SCRATCH/qwen3_real/models/Qwen3-30B-A3B/model.safetensors.index.json
$SCRATCH/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
$SCRATCH/openbookqa/validation.jsonl
```

## 方式一：Interactive 4小时

可先在登录节点运行 `tmux new -s cwhv3`，再申请。记录 `hostname`；重连 tmux 需要回到同一登录节点。tmux 只能防止终端断开，不能延长 Slurm 时限。

```bash
salloc -A m4431_g -C 'gpu&hbm80g' -q interactive -N 1 --gpus=4 -t 04:00:00
```

分配成功后确认设备：

```bash
srun -n 1 -G 4 --gpu-bind=none nvidia-smi --query-gpu=name,memory.total --format=csv
```

必须显示4张A100-80GB；不要把GPU attached memory的80GB与主机RAM混淆。

启动完整校准和核心对照：

```bash
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" focus v3a 220
```

220表示本次执行最多约3小时40分钟；脚本还读取当前作业 EndTime，最多运行到Slurm到期前三分钟。它在启动每个工作单元前检查剩余时间，并参考已完成单元耗时估计。估计不是绝对保证；硬中断也保留已经落盘的数据。

看到 `PAUSED_SAFELY` 属于正常保存后暂停。退出旧allocation，重新申请相同资源，仍然执行上面同一条 `focus v3a 220`。**保持tag为v3a；换tag代表另一个研究，不会接着旧结果跑。**

## 方式二：Regular 10小时，提交后可以关机断网

在登录节点直接提交，不需要先salloc：

```bash
sbatch -A m4431_g -C 'gpu&hbm80g' -q regular -N 1 --gpus=4 -t 10:00:00 \
  "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/nersc.sbatch" focus v3a
```

命令行10小时覆盖脚本中的默认12小时。脚本将依据真实Slurm EndTime留出三分钟停止余量。看到 `Submitted batch job JOBID` 后即可关闭本地电脑与网络，任务在集群继续排队/执行。计算节点故障、程序错误或达到时间上限仍可能中断，这些不能通过本地不断网保证。

`focus` 是完整模型、完整800请求×三次重复的核心研究，按旧速度估计可能在10小时内完成，但不能保证。若未完成，再提交**完全相同命令和tag**，自动恢复。

账户沿用你原来的 `m4431_g`；若项目账户变更，需要同时改命令行和脚本中的账户。

## focus具体运行什么

先冻结会话不重合的数据：Train800、Validation800、Diagnostic1600、Test1600。所有正式性能请求固定生成256Token、并发16；Short96–384输入Token，Long768–1536输入Token。

校准包含1600条全模型真实生成请求，训练隐藏状态用于实际Expert输出误差；48层质量评估和策略搜索都逐层保存。

核心正式对照：

| 模式 | 目的 |
|---|---|
| Native | 原生128槽位基线 |
| Selective CW | 相同 C/W/H 权重布局，副本不接单 |
| Selective Static | 相同布局，使用校准选定的固定p |
| Selective Dynamic | 相同布局，每次调用根据当前负载计算m |

4模式×2长度×800请求×3独立启动重复=**19200个正式性能请求、4915200输出Token**，另有逐模式OpenBookQA500题。它不是小模型或小数据试跑。

每个800请求性能单元使用一个新服务实例，避免中断前后的EPLB/缓存状态污染其它结果。QA也独立启动。冷加载会增加总时间，不能仅按生成吞吐量计算10小时是否足够。

## 断点保存的具体语义

| 部分 | 中断后处理 |
|---|---|
| 已完成的800请求性能单元 | 原样保留，检查指纹和校验和后跳过 |
| 尚未完成的性能单元 | 每个已完成请求的指标JSONL已flush/fsync，保留该次attempt；恢复时重测这一个800请求单元 |
| 已完成但尚未写done marker的单元 | 若完整JSON/CSV有效，可恢复为完成结果 |
| OpenBookQA | 每题落盘，从未成功题目继续 |
| Train/Validation校准 | 每个完整800请求单元分别提交；中断的那个重测，已完成的另一个不重跑 |
| C/W输出误差计算 | 每层原子保存，从尚未完成层继续 |
| 布局优化 | 每层保存partial，从尚未完成层继续 |
| Trace/Profiler | 已完成阶段跳过，未完成阶段保留旧attempt并重新采集该阶段 |

不能把中断前400请求和重启后400请求的walltime拼成800请求吞吐量。更不能从断开的某个GPU生成Token原位置恢复：本包没有保存CUDA context或KV cache。**恢复的是研究任务进度，不是GPU内存现场。**

所有正式文件在：

```text
$SCRATCH/cwh_qwen3_vllm/runs/cwh_v3_v3a/
  study_spec.json
  sessions/                 # 每次会话的GPU/节点/作业与状态
  calibration/              # 完整Train/Validation、逐层质量成本
  policies/                 # 固定的布局和动态策略
  formal/rep_0/...          # 三次重复，各模式的JSON/CSV/done标记
  diagnostic/...
```

`attempts/`下保留每次服务日志、请求日志、权重审计及失败原因，不覆盖旧日志。请求和问题日志逐条fsync；summary、逐层结果与done标记用原子替换。相同tag加独占锁，防止两个任务同时写；代码、模型索引、源数据或完成文件被改动时拒绝混合结果，需新tag。

## 监控、重连和恢复

登录节点可随时运行：

```bash
squeue -u "$USER"
python3 "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/tools/status.py" --tag v3a
```

普通任务日志：`CWH_v3_JOBID.out` / `.err`，客户端详细进度在每个attempt的 `client.log`。状态 `STAGE_COMPLETE focus` 表示核心流程完成；只有 `PAUSED` 则重跑同一命令。`FAILED` 要先看具体log，不要删除断言或伪造done文件。

在tmux中运行时，Ctrl+B再按D可detach；重新连到同一登录节点后 `tmux attach -t cwhv3`。若allocation还活着且任务仍执行，不要再同时启动一个同tag任务。若allocation已结束，重新salloc或sbatch后继续。

## 需要完整扩展实验时

只改stage，保持tag与代码不变：

```bash
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" formal v3a 220
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" diagnostic v3a 220
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" verify v3a 220
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" profile v3a 220
```

也可把`focus`改成`all`，自动依次处理校准、core、diagnostic、formal、Trace、Profiler；每次到时后重跑相同`all v3a`。core和formal共享检查点，核心数据不会再测一次。

formal完整9模式加入EPLB、频率优先、质量优先、原布局融合、静态Torch对照。formal43200性能请求+diagnostic48000=91200性能请求，纯生成按旧300–335tok/s粗算约19–22节点小时，另有加载、校准、QA等时间，**无法10小时全部跑完**。

完整formal的汇总可用：

```bash
python3 cwh_v3_bundle/tools/summarize_suite.py \
  --root runs/cwh_v3_v3a/formal --out runs/cwh_v3_v3a/formal_summary
```

focus只有四模式，`summarize_suite.py`也会统计当前四个全部完成的模式。三次重复仍只有同一500题，不能当作1500道独立质量题。

在暂停或完成后打包备份：

```bash
cd "$SCRATCH/cwh_qwen3_vllm"
tar -czf "$SCRATCH/cwh_v3_v3a_results.tar.gz" runs/cwh_v3_v3a
mkdir -p "$HOME/qwen3_research_archive"
cp "$SCRATCH/cwh_v3_v3a_results.tar.gz" "$HOME/qwen3_research_archive/"
```

然后可用scp下载。保留全套结果，包括负结果与未完成attempt。

## 参考与范围

资源和时限查阅：[NERSC队列](https://docs.nersc.gov/jobs/policy/)、[交互作业](https://docs.nersc.gov/jobs/interactive/)、[80GB约束与GPU请求](https://docs.nersc.gov/systems/perlmutter/running-jobs/)、[Slurm salloc](https://slurm.schedmd.com/salloc.html)。

动态部分按固定一个H副本的整数assignment min-max问题求闭式解；它不是考虑全部GPU kernel耗时、通信与所有质量任务的全局最优，也不是在线权重迁移。实际性能仍由这次GPU实验回答。
