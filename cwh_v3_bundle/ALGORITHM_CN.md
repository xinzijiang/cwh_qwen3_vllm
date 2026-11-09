# CWH v3：动态分流的公式、边界和选择依据

## 先定义“动态”

v2 的 C/W/H 和 p 均由离线校准确定。Token 经过 hash 进入不同副本不等于负载感知的动态控制。

v3 保留加载时的 C/W/H 槽位布局，**每个 MoE 层、每次真实调用都根据当前 Top-k assignments 重新决定转移多少个 H**。这是真正的在线分流；不声称实现在线 C/W/H 权重迁移。

如果 C 槽位已经装入 H，关闭 H 分流只能退回 CW，不能恢复原生 C。要恢复 C，必须恢复它的权重并同步所有 rank；该慢路径不在本版实现范围内。

## 一、为什么你的原始 idea 能成立

原生 MoE 输出：

\[
y(x)=\sum_{e\in S(x)}\alpha_e(x)f_e(x),\qquad |S(x)|=8.
\]

将选中的 C 交给 W，保留原 Router 权重时：

\[
\widetilde y-y=\mathbf1[C\in S(x)]\alpha_C(x)[f_W(x)-f_C(x)].
\]

这是近似误差，不是无损重定向。若 W 本身已在 Top-8 中，W 会出现两条 assignment，其两份加权贡献都要保留。

H 的副本使用真正相同的 BF16 权重，理想函数满足 f_replica=f_H。因此单独改变 H 的物理执行位置不改变上述数学输出；实际 BF16 路径仍可能有数值差异，需要 GPU 检查与实际质量验证。

本方法用 C→W 的质量代价换取一个 H 执行槽位，128 个原物理 Expert 槽位保持不变。

## 二、离线质量约束与联合布局

使用真实校准隐藏状态，对每个 C/W 对计算：

\[
R(C,W)=\sqrt{\frac{\sum_x\|\mathbf1[C\in S(x)]\alpha_C(x)(f_W(x)-f_C(x))\|_2^2}
{\sum_x\|y^{Native}(x)\|_2^2+\epsilon}}.
\]

含义：替换引起的输出扰动，相对于整个 MoE 输出有多大。它比单独看 Expert cosine 更符合保留 Router 权重的 C→W 语义，但不是跨层任务 Accuracy 的误差上界。

实现选择低频 C 候选，对所有 128 个 W 测量真实 SwiGLU 输出，再按 GPU 保留两个误差最小的 W。H 从各 GPU 的高频候选中选。Train 中选一个 C/W/H 组合，Validation 只决定接收或拒绝；不反复换候选来适配 Validation。

联合目标是：

\[
\min_{C,W,H}\mathbb E_t[\max_g \widetilde L_{t,g}(m_t)]
\quad\text{s.t.}\quad R(C,W)\le\varepsilon.
\]

L 是 assignments 成本代理，不是真实 GPU 时间。实现还要求 Full 对 Native 与同 C/W 的 CW 均有稳健代理改善。候选池有边界，不能称为整个 Qwen3 模型上的全局最优。

## 三、在线闭式分流

以下所有负载均先执行 C→W 后再计算，不能用原 Native 负载来决定分流。

令 s 为 H 原 GPU，d 为 C 槽位所在 GPU：

- a=L_s：源 GPU 当前 assignments 数。
- b=L_d：副本 GPU 当前 assignments 数。
- h：当前真正选中 H 的 assignments 数。
- o=max_{g≠s,d} L_g：另两张 GPU 的最大负载。
- m：交给副本的 H 数量，0≤m≤h，且是整数。

目标：

\[
\min_{m\in\{0,\ldots,h\}}F(m),\qquad
F(m)=\max(a-m,b+m,o).
\]

固定这一份副本和等成本 assignments 假设时，最小可达峰值为：

\[
\boxed{M^*=\max\left(o,\left\lceil\frac{a+b}{2}\right\rceil,a-h,b\right)}.
\]

在所有达到该峰值的解中，移动最少的那个为：

\[
\boxed{m^*=\max(0,a-M^*)}.
\]

为什么：四项分别是其它 GPU 的固定峰值、两张 GPU 的总工作量下界、H 全部转出后的源负载下界，以及目标 GPU 无法减少的原负载。满足这些下界后，只需转出 a−M* 个，就能让源不超过 M*；两张 GPU 总量约束保证目标也不超过 M*。这也证明该 m 移动最少。

例子：a=50，b=30，o=44，h=20。

\[
M^*=\max(44,40,30,30)=44,\qquad m^*=6.
\]

转移 6 个后，源44、目标36，整体峰值44。转移10个也会让这两张卡都到40，但另一个GPU仍是44；多移动的4个没有降低峰值。

若 CW 后四卡为 [40,56,22,36]，H 在GPU0、副本在GPU2，则 o=56，m*=0。这个副本无法缓解GPU1，本次分流就关闭。这正对应你原方案经常没有缓解最大瓶颈的问题。

另一个 Batch 的源可能更重、目标可能更轻，同一个 H 副本便会得到不同的 m。这里不需要假定固定25%或50%。可报告当次 p=m/h（h=0时定义0）。

## 四、防止小收益支付额外开销

代码默认要求：峰值至少降低1个 assignment，且代理改善至少1%，否则 m=0。

\[
m=\begin{cases}
m^*, & L_{max}-M^*\ge1\ \land\ \frac{L_{max}-M^*}{L_{max}}\ge0.01,\\
0, & \text{其它情况}.
\end{cases}
\]

这是开销保护启发式。闭式解在加这个保护之前是该整数代理问题的精确最优；加入保护后，算法有意拒绝很小的代理收益。1%不是已经测得的 GPU 盈亏平衡点，必须用 Profiler 和实际吞吐量验证。

本版动态控制仍要运行计数和决策，即便最后 m=0；关闭分流不是零路由开销。

## 五、GPU 实现

当前固定为已经验证路由输入会聚合的 vLLM 0.13.0 / Ag-Rs / EP4 路径。每个 Worker 应看到相同完整路由，校准用四 rank 指纹核验；换成其它后端可能需要全局计数 collective，不能直接复用这里的假设。

小 Batch（最多4096个 assignments）用一个 Triton CTA 完成 C→W、四 GPU 计数、闭式 quota 和 H 的前缀编号派发，只增加一次 launch。大 Batch 用一个部分计数 kernel 和一个全局前缀重定向 kernel，共两次 launch。

通过全局 H ordinal 恰好派发 m 个 H，保留每条 assignment 的原权重。热路径没有 CPU `.item()`、CPU tensor 传输或额外 NCCL collective。调试用 Torch/CPU oracle 只用于正确性，不能拿来当性能 arm。

## 六、与相关工作的关系

| 来源 | 本方案借鉴 | 保留的自身问题 |
|---|---|---|
| [CRAFT](https://proceedings.mlsys.org/paper_files/paper/2026/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) | 逐层副本收益和预算意识 | 将 C→W 的质量代价与回收原槽位一起考虑 |
| [SERE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d07eb2335d88b1dfcefcc3d2da2a230-Abstract-Conference.html) | 相似 Expert 重路由 | 回收出的槽位实际承载 H 真副本 |
| [R&Q](https://arxiv.org/abs/2602.19938) | 原内存预算内复制的目标 | 保持 BF16，用 C→W 腾槽位，不采用量化路径 |
| [LPLB](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) | 当前 Batch 的 min-max 副本分流 | 对固定一个 H 副本的特殊情况使用闭式整数解 |

这是针对当前实现的工程选择，不是复现上述全部系统，也不是证明已经优于所有现有方法。完整检索不可能保证覆盖市面全部方案。固定副本位置、等成本 assignments 的局部最优，不能冒充考虑 HBM、GEMM padding、通信、多个副本和所有质量任务的全局最优。

建议的成功标准是：Dynamic 优于相同布局的静态 p；Dynamic 优于相同布局的 CW；固定128槽位下优于Native；在事先约定的独立质量任务中可接受。如果只是动态调了 m 但没有净加速，应该报告负结果。
