# CWH 原代码审计、算法改进与研究定位

依据：用户提供的 `cwh_paper_review_60706450_20261102_122326.tar.gz` 内真实源码、24 份性能 JSON、4 份逐题质量 CSV、四个 Worker 的路由日志及策略文件；同时参考现有 vLLM 0.13.0 官方文档和下列研究原文。不能确认远端之后未上传的代码修改。

## 1. 最重要的结论

你的核心机制已经实现：原 C 的 checkpoint 张量被跳过；原 H 的 gate/up/down 权重通过 loader 加载到 C 的现有物理槽位；路由将 C 交给 W，将部分 H 交给该副本。这个实现没有新增 Expert 物理参数槽位，也不是只有一个概念图。

当前失败的是选取与执行策略，而不是“副本完全没工作”。原 H/C/W 的 GPU 角色固定，H 分流取决于两列无负载含义的 Router Logit，校准仅来自 8 条 Prefill Prompt，且热路径需要多个小 GPU 操作。

应该继续研究的对象是：**一笔有质量代价的 Expert 槽位交换，能否带来足够大的关键路径净收益。** C、W、H、目标 GPU 与分流比例必须一起考虑。

## 2. 重算旧结果

| 模式 | Short output tok/s | Long output tok/s | OBQA |
|---|---:|---:|---:|
| Native EP | 331.83 ± 4.52 | 335.26 ± 3.14 | 405/500 |
| EPLB-4 | 269.93 ± 0.61 | 272.15 ± 6.19 | 410/500 |
| CW-only | 319.99 ± 0.75 | 321.68 ± 1.74 | 419/500 |
| Full CWH | 294.88 ± 2.22 | 305.28 ± 1.08 | 418/500 |

均值±样本标准差，各 3 次；每次 800 请求、256 输出 Token。Full 对 CW 分别下降 7.85% / 5.10%，对 Native 下降 11.14% / 8.94%。原来的“超过 EPLB”不能代表超过 Native，也不能代表超过调优后的 EPLB。

原 EPLB 使用 window=256、interval=100 的激进同步重平衡配置。新的对照采用 vLLM 0.13.0 文档列出的 window=1000、interval=3000，并明确同步模式；这仍是一组配置，不是最优 EPLB 的证明。EPLB 的 4 个冗余槽位使每层成为 132 槽位，Native/CWH 是 128。见 [vLLM 0.13.0 parallel config](https://docs.vllm.ai/en/v0.13.0/api/vllm/config/parallel/)。

质量方面，CW 和 Full 有 15 道预测不同，但净正确率只差一道；不能用 Accuracy 接近证明执行等价。Native 和 CW 有 32 道预测不同，Native 单独答对 4 道，CW 单独答对 18 道；精确配对 McNemar p≈0.00434（未做多重比较校正）。所以在这份固定评测上，观察到的提高有配对证据，不能一概归为随机波动；但它依然不能证明更广泛任务上的质量提升，更不能把 500 题结果当无损保证。

完整重算数据见 `ORIGINAL_RESULTS_RECOMPUTED.json`。旧实验同一 arm 的三轮共用服务实例，不等价于三次独立模型启动；新实验改为三次独立启动块。

## 3. 必须纠正旧评估的一点

旧 Trace 的 4,848 组 layer/call，每组确实存在四个 Worker 记录。重算检查发现：**四份记录去掉 PID 后完全相同**。因此这次采用一个 Worker 去重有原始证据，不只是猜测。

然而，不能从长时间聚合的 GPU 计数推出逐次关键路径：

\[
\max_g\sum_t L_{t,g}\ne\sum_t\max_g L_{t,g}.
\]

GPU 轮流过载时，左边可能非常均衡，右边仍然很大。

在旧 Trace 的 16 行小 Batch 样本中：

| 代理指标 | Native | CW 反事实 | Full |
|---|---:|---:|---:|
| 聚合 Skew | 1.0264 | 1.0475 | 1.0475 |
| 每层每调用 Skew 的平均 | 1.2445 | 1.2468 | 1.2449 |
| 每层每调用最大 assignments 的平均 | 39.8231 | 39.8992 | 39.8383 |

因此“Native 只有约 2.6% 可优化空间”不能作为这个实验的可靠上限。逐调用的负载峰值被平均掩盖了。同时，这不意味着已经有 24.45% 的端到端提速空间：所有这些都是 assignment 代理，GEMM tile、活跃 Expert 数、HBM、通信与其它层都可能限制收益。

Full 相对相同输入上的 CW 反事实，最大 assignments 均值只改善约 **0.152%**。这比“聚合最大 GPU 完全没变，因此 Replica 没有任何帮助”更准确：它有很小的逐调用代理改善，尚未形成端到端净收益。CW 反事实来自 Full 轨迹的同一批输入，并不是另一次 CW 运行；由于上游激活会改变，不能当独立 CW 服务的真实全部负载。

旧 Trace 没有可靠的 scheduler phase 字段。902 行和16行分组与 Prefill/Decode 的解释相符，但仅从行数不能严谨给所有调用标相位。新代码从 FlashAttention 的 query 长度提取相位，并固定关闭 chunked prefill。

## 4. 算法与工程审计

| 发现 | 是否一定是 bug | 改进 |
|---|---|---|
| H0/W1/C2 固定 | 不一定导致错算，但限制算法解空间 | 按实际 owner 泛化，检查完整 expert_map 与同步权重加载 |
| `(hot+13)` 与 `(hot+37)` Logit 比较 | 分流能发生，但无负载目标且比例失控 | 旧规则留作对照；新规则采用可复现的期望比例 p |
| 每层多次 compare/where/zeros_like、环境变量读取 | 可能造成开销，不能只凭源码量化耗时 | 单 kernel 后处理；Shadow 和同语义 Torch/Fused 对照 |
| C→W 保留原 C 的 Router 权重 | 正是该近似算法的语义，并非天然无损 | 用实际加权输出误差评估 C/W，做独立任务质量验证 |
| W 已在 Top-8 时出现重复 ID | 不自动等于错误；两条权重的贡献应相加 | 保留两个位置；使用实际 vLLM kernel 测试，不能静默删项 |
| H 权重 alias 加载计数 | 能证明张量流结构，不能证明实际字节相等 | 新增源 H 与目标槽位 w13/w2 SHA256 审计 |
| Full 和 CW 预测不完全相同 | 可能是数值路径、排序或实际错误 | 权重哈希+低层副本检查+逐题选项 logprob；必要时完整激活比对 |
| 关闭 Replica 即声称恢复 Native | **若 C 已被覆盖，这样做会错误计算 C** | Native fallback 必须在加载前保留原 C；运行时 p=0 仍须 C→W |

全套方案是静态加载时槽位回收，不是在线 malloc/free，也不减少总 Top-8 assignments。Replica 不保证减少通信：在你现在的 Ag/Rs 后端尤其不能按迁移 assignments 数直接推算增加了多少网络字节。

## 5. 相近工作与可研究差异

这不是“尚无人做 Expert 复制”的领域。以下是本次查阅的主要原始来源；对没有读到完整实现的来源只使用其明确描述，不推断其所有实验结论。

| 工作 | 已有主意 | 对你的意义 |
|---|---|---|
| [LLEP](https://proceedings.mlr.press/v306/nguyen26m.html) | 将过载设备的 Token 和相应 Expert 参数工作转到较轻设备 | 说明保留逻辑 Expert 身份的负载转移已有成熟研究 |
| [CRAFT，MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) | 按收益与内存预算进行细粒度逐层副本分配 | “只在有收益的层复制”本身不能当新颖贡献 |
| [SERE，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d07eb2335d88b1dfcefcc3d2da2a230-Abstract-Conference.html) | 以相似 Expert 重路由减少 Decode 活跃 Expert，并提供高效 kernel | “相似替换”或“融合路由”也不能单独当新意 |
| [R&Q](https://arxiv.org/abs/2602.19938) | 量化较不重要的 Expert 与副本，在原内存预算内复制热门 Expert | 与“腾内存换复制”高度相近；你的路线保持 BF16，通过 C→W 回收原槽位 |
| [LPLB，作者技术文](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) | 用实时计数优化已有副本的逐 Batch min-max 分流 | “根据实时负载选择分流比例”也已有实现 |

CRAFT 原文还使用逐 Batch 回放估计收益，因此不能把“使用回放而非平均热度”单独包装成新方法。LPLB 保持逻辑 Top-k 身份，你的 C→W 是带近似质量代价的交换，研究约束不同。以上比较不证明你的完整组合绝无先例；论文新颖性仍需继续检索、实现与实验支持。

## 6. 建议聚焦的新问题：联合评估一笔槽位交换

建议研究表述：**在固定 BF16 Expert 槽位预算下，联合优化替换质量代价与热门副本的关键路径收益。**

一个候选不是单独的 C 或 H，而是：

\[
\theta_\ell=(C_\ell,W_\ell,H_\ell,p_\ell).
\]

为每层和每个真实调用 t，设 Expert 频次为 n、GPU 原负载为 L，owner 为 g，则：

\[
\widetilde L_{t,g}=L_{t,g}
-n_{t,C}\mathbf1[g=g_C]
+n_{t,C}\mathbf1[g=g_W]
-r_t(H,p)\mathbf1[g=g_H]
+r_t(H,p)\mathbf1[g=g_C].
\]

r 是该调用真实 hash 分流产生的 H 数量，不是直接以 p×H 替代整数小 Batch 的实际行为。

当前实现的目标代理：

\[
J(\theta)=\mathbb E_{t\in Decode}\left[\max_g\widetilde L_{t,g}\right].
\]

质量代价用原始隐藏状态上的真实预训练 SwiGLU 输出计算。对于保留原 Router 权重 α 的 C→W：

\[
\Delta y_t=\mathbf1[C\in TopK_t]\alpha_{t,C}
\left(f_W(x_t)-f_C(x_t)\right),
\qquad
R(C,W)=\sqrt{\frac{\sum_t\|\Delta y_t\|^2}{\sum_t\|y_t^{Native}\|^2+\epsilon}}.
\]

相比只看 Expert cosine，这个指标同时包含 Router 权重、是否真的选中 C，以及整体 MoE 输出的尺度。它是固定层输入上的精确替换差的数值估计，不是跨层质量退化上界；小的 R 也不能自动保证所有任务安全。

真正需要证实的贡献假设是：**最冷的 C、最相似的 W、最热的 H 分别选出来，往往不能组合成最值得执行的槽位交换；联合选择能在相同质量约束下取得更好的速度。** 本包加入 Frequency / Quality 两个解耦对照，正是为了检验这个假设。

W 可以和 C 在同一张 GPU，也可以在其它 GPU。若同卡 W 的质量足够接近，它能避免强制把 C 工作转给第三张 GPU；但不会强行规定同卡，因为某层更好的解可能确实需要跨卡 W。

## 7. v2 已实现的部分

1. **完整模型、真实 Decode 校准。** Train/Validation/Test 会话隔离；记录每调用路由，按相位建立计数和整型分流回放。训练隐藏状态只用于 C/W 误差估计。
2. **质量与负载联合选取。** C 从每 GPU 最冷两个且低于平均频率 0.75× 的候选选；W 评估全部 128 个预训练 Expert，保留每 GPU 误差最小两个；H 候选为每 GPU 最高频三个；p∈{.25,.5,.75,1}。这是有边界的候选搜索，不能宣称全局最优。
3. **Replica 的独立增益门槛。** Full 既要优于 Native，也必须优于同 C/W 的 CW-only。不能把 C→W 本身改变负载的收益冒充 H 副本收益。
4. **验证集准入。** 每层只拿 Train 选中的一个候选去 Validation 验证，没过则保持 Native；不在 Validation 上逐个试候选挑赢家，更不用最终 Test 选策略。
5. **低开销执行。** 在原 `fused_topk` 后，以单个 Triton kernel 同时做 C→W 与 H→C，不改 Top-k 权重、不原地覆盖原 IDs。它只融合路由后处理，没有声称融合 Top-k、通信与 GEMM。Token 数使用运行时参数，避免每种 Prefill 长度都编译一个新 kernel。
6. **严格的零额外 Expert 槽位。** 未启用层保留 C；启用层将 C 槽位交给 H；CW-only 与 Full 使用完全相同的权重布局。离线质量计算的临时缓存和路由 scratch 不应混称为额外 resident Expert 参数。

新 hash 不使用任意两列语义 Logit 决定分流。它对 Token 行号和调用编号进行可复现混合，p 是期望分流比例；各 Worker 输入与调用对齐由指纹检查保障。该规则仍不是完全动态的在线负载控制器；它是为了可靠地执行离线选定的 p，并暴露可验证的消融变量。

## 8. 下一阶段真正值得加深的主意

先验证上述联合交换，再优先做 **“H 成为瓶颈时，哪个 C 槽位最有空闲价值”**，不要继续只追更高的平均 Hot 频率。

可记录两个条件统计：

\[
P(g_H=\arg\max_g L_{t,g}\mid H\text{ 被选中}),
\qquad
\mathbb E[(L_{t,g_H}-L_{t,g_C})_+\mid H\text{ 被选中}].
\]

某个 GPU 的长期平均负载低，不代表 H 真正过载时它也空闲。若 H 和目标 GPU 的负载峰值同步，副本可能没有价值；如果峰值互补，同样的平均热度也可能获得更大收益。本包逐调用 replay 已部分考虑这个非线性关系；将它显式用于候选筛选、漂移鲁棒性和可解释性，是继续深化的研究方向，不能宣称已经完成了在线控制实现。

随后才把代理目标替换为实测成本：

\[
\mathbb E_t[\max_g T^{Expert}_{\ell,t,g}]
+T^{Route}_{\ell}+T^{Comm}_{\ell}.
\]

必须用 Profiler 判断问题到底是 token 数、活跃 Expert 数、tile padding、HBM 读取还是 collective。副本可能降低 Token 数却增加目标 GPU 的活跃 Expert 集合和权重读取，特别是在 Decode；因此 assignments 改善不保证 GEMM 改善。这一步应基于本包的 Profiler / 真实 Trace 结果建立成本模型，不要现在给未经测量的微秒数。

## 9. 实验顺序与决策

| 阶段 | 体量 | 必须回答 |
|---|---|---|
| 原结果 CPU 重算 | 已完成，无 GPU | 旧结论和 Trace 是否可信 |
| 新校准 | 800 Train + 800 Validation，全模型、各256输出 | 是否存在稳健的 C/W/H/p 候选 |
| 固定布局诊断 | 10 arm、48,000 性能请求、三次独立启动 | 开销、旧分流规则和分流比例各有什么影响 |
| 新算法正式对照 | 8 arm、38,400 性能请求、独立 Test | 联合选择是否优于解耦选择、CW 与 Native |
| 运行机制验证 | CW / Full 各800真实请求 + 三模式 Profiler窗口 | Replica 是否真的改善每层执行而不只是改计数 |
| 后续通用性 | 先冻结候选再扩 concurrency=8/32、工作域与至少两个独立质量任务 | 何时适用、何时失效、质量是否可接受 |

没有把主研究改成量化、跨层替换、改训练 Router 或单纯增加 GPU 数。当前也不建议直接实现每步在线权重迁移：一旦 C 被覆盖，完整 Native 恢复需要恢复权重、同步 rank 和承担迁移成本，会把因果问题混在一起。

成功证据应同时包含：在同布局下 Full 比 CW 快；联合策略比相同约束下的独立选择更好；固定 128 槽位下胜过 Native；质量在预先声明的多任务容忍范围内。若只在强偏斜的真实域有收益，就报告这个适用边界，不把它包装成所有 Workload 都有效。

## 10. 验证状态

已完成：真实归档重算；四 Worker 4,848 组记录一致性检查；7 项 CPU 测试，包含槽位流式替换/禁用、非固定 owner、重复 ID 语义、hash 回放、聚合陷阱、流式 HTTP Token 数检查，以及训练改善但验证漂移时拒绝候选的集成测试；所有 Python 文件编译与两个 Shell 脚本语法检查。

未完成：Triton 实际编译、CUDA ID 对照、vLLM GPU 数值门禁、真实 checkpoint 新加载流程、真实校准的输出误差计算、新算法吞吐量和质量实验。它们都需要用户的目标 GPU 环境；不能声称新算法已加速、已无损或已达到论文新颖性要求。
