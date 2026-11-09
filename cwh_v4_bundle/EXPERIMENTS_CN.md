# 实验计划与组员建议落实

## 核心问题

主结论必须是“完整 C→W 槽位回收 + H Replica + 动态决策”相对 Native 是否提高服务吞吐，且质量损失是否可接受。CW-only 仍保留为诊断消融，用于判断近似替代本身的代价；它不能替代主结论。模型输出一致性不靠“同一个 H”口头保证：加载后逐 rank 比较 H 与 replica 的 fused weight SHA256，路由算法对照 CPU oracle，实际质量另测。

组员提到 kernel local-slot compaction、避免处理非本卡 expert 和 launch 开销，这是独立而相关的系统优化。没有拿到他们的源代码和逐项 profiler 证据，不能把其口述的加速比例直接移植到本算法，也不能声称 V4 已实现那个 kernel。后续拿到实现后做2×2：原kernel/组员kernel × Native/CWH，检验组合增益与交互项。当前 profile 阶段用于提供数据接口和瓶颈证据。

## 已实现矩阵

| Stage | Arms | 重复/工作负载 | 目的 |
|---|---|---|---|
| check | CUDA oracle、加载前环境检查 | 无正式性能 | 在跑长任务前发现环境/API/路由错误 |
| calibrate | Native probe、质量误差、硬件cost、逐层实测veto | train/validation分离 | 选择并冻结 C/W/H |
| pilot | Native、新cost | 2×short/long×800；diagnostic split | 测实际净收益，产生采用建议 |
| main | Native、新cost、EPLB-0、EPLB-4 | 3×short/long×800；每arm/repeat 500QA | 正式完整方法比较 |
| ablation | Native、identity callback、CW、static、old_dynamic、cost_shadow、cost | 3×short/long+QA；diagnostic | 同放置比较决策与执行开销 |
| regression | Native、v3_count static/dynamic、cost | 3×short/long+QA；diagnostic | 比较旧/新选择与调度组合 |
| sweep | Native、cost、static | concurrency 1/4/16/32/64，各3轮+QA | 看不同服务负载下转折点 |
| profile | Native、identity、CW、static、old_dynamic、cost | 短小单独profile | 按kernel/通信/路由拆分；不报正式速度 |

regression 的 v3_count 由旧 planner 在同一新校准数据上生成，不是恢复历史JSON的精确复现。不要把它写成历史 v3c 同一次实验。

## 公平性

所有主比较固定4×A100-80GB、BF16、TP4/EP4、同容器、同tokenizer、同attention/backend、同eager模式、同KV blocks，禁止拿40GB卡结果直接相比。concurrency=16指最多16个并发请求，不保证每次MoE都正好16 tokens；实际调用包含prefill/mixed/padding。

short输入96–384 tokens；long输入768–1536；每个性能请求强制256个output tokens；每轮800请求，不能通过更早EOS获得表面优势。train、validation、diagnostic、test按conversation去重分离。

每个arm、repeat、workload独立启动服务，固定相同warmup：8轮，每轮max(32,2×concurrency)请求，每个生成256tokens。这样减少EPLB冷启动未适应的偏差；不能只给新算法warmup。EPLB配置window1000/interval3000/同步更新；记录服务日志，检查实际适应情况，不能仅凭warmup请求数证明每个worker一定更新了几次。arm顺序轮换并交错，以降低顺序偏差。

Native vs identity 可识别自定义routing callback是否换了上游路径。identity在48层安装callback，因此是全层callback诊断，和≤8层CWH的开销不能逐项直接相减。cost_shadow 小调用计算相同决策并保留diagnostic写入，最终只做CW；其额外scalar store意味着它是接近的诊断，不是严格零差异反事实。大调用不做动态cost，shadow走CW，不能用它估计大调用static mask完整开销。

硬件cost/本地replay不包含collectives；正式服务有真实通信。profiler与capture会扰动速度，单独跑。固定KV预算用于公平性，不等于调优后的最高生产吞吐；最终可在结果清楚后另做CUDA Graphs研究，本版本不声称支持graph。

## 记录和判定

主指标 output tokens/s；辅指标TTFT、TPOT p50/p95、请求错误数、输出token总数、显存及权重槽位数。记录每层active专家数/tiles代理/移动数量可从calibration回放推算；正式速度路径不逐步CPU同步打印。

统计单元是独立服务启动重复，不是800个请求。吞吐采用同rep相对Native的log ratio、三次重复t区间；3次重复区间可能很宽，应如实报告。若interval跨1，不能宣称稳定加速；若要追加重复，需要另行预先确定试验计划，不能一直跑到显著。

OpenBookQA确实包含：main/ablation/regression/sweep都有500题评估；pilot/calibrate没有正式QA。保留每题预测，做paired差异与McNemar检验。三次相同500题不是1500个独立样本；质量比较按每轮配对呈现。已有500题的历史结果已经看过，因此这里只能作为质量回归集，不能包装成完全未见过的最终泛化测试。后续可另行增加未用于选择的官方test/其它任务，当前不自动下载或假装已有数据。

预先设定工程目标：held-out short/long吞吐比Native高，p95TPOT退化≤5%，QA准确率下降不超过1个百分点。QA“≤1点”的点估计不是非劣效证明；应同时报告配对区间、错误改变题目，并承认500题证据有限。

## 结果驱动的下一步

- 0层通过：报告此硬件/负载没有足够收益，保持Native；检查cost拟合误差与质量覆盖，不能删除门槛强行宣称成功。
- cost≈shadow且都慢于CW：路由/调度开销重要；优先研究上游topk融合或已测有效的静态规则。
- shadow≈CW、cost更慢：检查新激活expert、tile padding、通信及热点GPU是否转移。
- static优于cost：保留static为更优部署候选，报告动态计算的代价；“动态”不应成为强制目标。
- 新cost优于旧dynamic但仍低于Native：只能说修复了部分退化，不能说实现加速。
- 质量下降明显：减少启用层、增加C条件样本、收紧误差预算；使用新tag和新的预定实验，不用测试题答案挑W。
- 并发升高才获益：将结论限定到该服务负载，不宣称通用收益。

## 已针对历史bug修改

| 历史问题 | V4处理 |
|---|---|
| 宿主3.6无datetime.fromisoformat | time.strptime + time.mktime；按真实Slurm EndTime截止 |
| U+2028导致JSONL被拆成多行 | 按LF/逐文件行读取，并忽略空行；Unicode回归测试 |
| Address already in use | 每个单元重新选择空闲端口，SO_REUSEADDR探测，退出清理所属进程组 |
| runtime monkeypatch/runpy工具import路径 | 从正常tools入口启动，不依赖临时wrapper |
| 老PYTHONPATH/vendor污染 | PYTHONPATH固定新包；清理旧路由环境变量 |
| Python/torch/vLLM混用 | 容器统一执行，启动前检查版本、GPU内存、model和tokenizer |
| 断线后重复已完成测试 | 原子done+hash+文件锁；完成轮次跳过，半轮保留日志重做 |
| 源代码中途改动混合实验 | study_spec冻结代码/数据/环境；变化拒绝同tag续跑 |
| 副本权重或EP map不匹配 | 精确linear expert map检查、流式权重alias计数、各rank tensor hash |

无法在交付机远程保证Perlmutter CUDA行为；本包GPU自检与首个真实服务启动是必要验证，不应跳过。端口空闲探测与实际bind之间仍可能发生外部竞争；发生时日志会保留，下一次同tag用新端口重试，不删除成功结果。
