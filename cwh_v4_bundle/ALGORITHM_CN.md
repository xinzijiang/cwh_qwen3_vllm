# 算法说明：固定槽位预算下的成本感知 CWH

## 为什么旧 dynamic 可能更慢

旧目标约为 min max_g N_g，其中 N_g 是一张卡接收的 assignments 数量。它隐含每个 assignment 成本相同。Decode 中一个 expert 只服务很少 tokens，权重读取、启动 kernel 和小矩阵 padding 可能比算术量更重要。把 H 分成两份，可能让两张卡同时读取 H 权重，两个小 GEMM 都支付启动/填充开销。因此计数更平衡，不等于速度更快。

当前已有结果足以否定“旧 dynamic 必然加速”的判断，但没有完整的新 dynamic 原始日志，不能断言其减速全部来自某一种 kernel。V4 提供同放置的 old_dynamic/static/cost/CW/shadow 对照，定位到底是决策计算、放置、近似替代还是 expert 执行导致。

你的贡献应该围绕：**用低损失 C→W 近似回收固定专家槽位，为 H 提供副本，并在收益足以覆盖代价时使用它。** Replica 本身不是新贡献，dynamic 这个词也不是贡献。

## 1. 质量约束

原始 MoE 输出：

$$y_t=\sum_{e\in\mathrm{TopK}(t)}p_{t,e}f_e(x_t).$$

替换 C 后的局部变化：

$$\Delta y_t=\mathbf1[C\in\mathrm{TopK}(t)]p_{t,C}(f_W(x_t)-f_C(x_t)).$$

用校准 hidden states 和真实 checkpoint 的 SwiGLU 计算：

$$E_{C,W}=\sqrt{\frac{\sum_t\|\Delta y_t\|_2^2}{\sum_t\|y_t\|_2^2+\epsilon}}.$$

这是“router 权重加权的整层输出 RMS 相对误差”，比只比较 expert 权重 cosine 更贴合替代操作。默认要求 E≤0.05、C 的 Decode routing mass≤1%；每层每卡从两个低频 C 开始，W 搜索所有 128 个 expert，保留每张卡两个误差最小的 W。至少需要16个 C 条件样本，否则不信任它。

误差是 train 局部代理，不能保证下游生成/QA 不受影响。最多8层、误差代理和≤0.15是保守工程预算，不是端到端误差上界。若 C/W 同时在 Top8，保留两项及各自权重，不能去重后丢掉贡献。

## 2. 硬件成本代理

每次调用，对 expert e 计数 n_e。对于 GPU g：

$$A_g=\sum_{e\in g}\mathbf1[n_e>0],\quad B_g=\sum_{e\in g}\lceil n_e/b\rceil,\quad N_g=\sum_{e\in g}n_e.$$

A 是活跃 expert 数；B 是按有效块大小 b 估计的 padded 工作块；N 是 assignments 总数。按 token rows 的2次幂 bucket，拟合非负模型：

$$\widehat T_g=d+aA_g+uB_g+vN_g.$$

a/u/v/d 来自四张 A100 上真实形状的 vLLM fused_experts CUDA event 测时。权重/激活随机 BF16，路由形状取 train 数据并补充合成形状。拟合使用训练形状和保留形状，b 从8/16/32/64中选择。b 是有效特征参数，不能声称等于所有 vLLM kernel 实际 BLOCK_M。p95 相对误差>25%的 bucket 不参与策略接受。25%误差远大于小幅收益，因此下面还有独立实测否决，不能仅凭拟合3%就宣称获胜。

该模型不含 attention、跨卡通信、CPU调度、缓存干扰或完整模型多层串行开销，且四卡共用拟合系数。需要端到端 pilot/main 检验。

## 3. 在线分流

先构造 C→W 后计数：n_W←n_W+n_C，n_C←0。对 H 当前数量 n_H 比较：

$$\mathcal M=\{0,\lfloor n_H/4\rfloor,\lfloor n_H/2\rfloor,\lfloor3n_H/4\rfloor,n_H\}.$$

对于 m，将 H 原位置计数减 m、C 副本位置加 m：

$$m^*=\arg\min_{m\in\mathcal M}\max_g\widehat T_g(m).$$

相同成本优先更小 m。若收益没有超过 max(固定 margin, C→W 基线成本的1%)，取 m=0。100%这个候选很重要：它允许把本次调用的 H 全部移走，关闭源卡 H 的活跃开销，而不是永远把一个 expert 拆成两份。

直方图、五候选成本、门槛、前缀计数、expert ID 重映射融合为一个 post-topk Triton kernel。最多512 rows，无 CPU readback、无额外 NCCL、无在线权重复制。它仍然增加一个 kernel launch，不能称为“零开销”或已经与上游 topk/GEMM 完全融合。

这只是在五个可选配额中的代理最优，不是全局最优。超过512 rows时使用 train 选择的固定0/50/100%比例。每个 H assignment 按确定顺序选择，四个 rank 必须有相同 gathered 路由输入，capture 指纹会检查这个前提。

## 4. 离线放置及四道筛选

1. 在 train 上按质量筛选 C/W，然后在每卡最高频 H 中搜索跨卡副本，联合选择大调用静态比例；不再固定 GPU0/1/2 角色。
2. 对每层只把 train 的最佳候选交给 validation；要求 Decode 代理至少3%改善、prefill/mixed退化≤1%、Decode逐调用p95退化≤5%。按收益排序最多8层且受质量预算限制。
3. 对通过的层，用 validation 的最多6个纯Decode、2个prefill/mixed真实路由形状，在四GPU逐卡测量 Native 与“实际路由kernel+fused_experts”。三次交错测量都需 Decode至少2%收益、prefill/mixed退化≤1%、Decode p95退化≤5%。没有通过就否决，不在 validation 上再换另一个候选。此回放是局部计算代理，不含collectives，不能冒充四卡端到端延迟。
4. 诊断 split 端到端 pilot：两次独立重启、short/long各800请求。每次都需≥2%吞吐提升且p95TPOT退化≤5%。报告候选但不隐藏负结果。正式main另用test split并做QA。

在线 margin 比较的是相对 CW 的分流收益；离线筛选和正式结论比较的是相对完整 Native 的净收益，二者不能混淆。

## 5. 与相关工作的关系

以下是最相关方法，不代表穷尽全部文献或实现了它们的官方代码：

- METRO：强调 memory-bound 时应关注 activated experts，token balancing 可能反而更慢。本实现采用 active-expert 成本特征，但研究重点是先通过近似替代腾出固定槽位。https://arxiv.org/abs/2512.09277
- TEMPO：研究混合 memory/compute 机制下的 makespan-aware routing，提示不能只看单一计数。本实现是小候选网格、简单非负模型，不是 TEMPO 官方实现，不宣称首次提出成本感知路由。https://arxiv.org/abs/2608.13057
- CRAFT：按层考虑复制收益与内存代价；本实现借鉴选择性启用，不在每层强制添加副本。MLSys 2026 官方论文页可按 CRAFT 检索。
- R&Q：强调 expert 热度与重要性并不相同，以量化提供复制空间；本项目保持 BF16，用 C→W 近似回收槽位，资源来源不同。https://arxiv.org/abs/2602.19938
- vLLM EPLB：必须单独控制冗余槽位。EPLB-0与本方法同128槽位；EPLB-4每层132槽位，不能当作完全相同专家内存预算。https://docs.vllm.ai/en/v0.13.0/serving/expert_parallel_deployment/

以上启发支持修改方向，不证明我们的实现优于这些方法。论文比较需要同硬件、同模型、同工作负载的可复现实现；当前包仅实现vLLM EPLB基线、旧计数dynamic和新CWH。
