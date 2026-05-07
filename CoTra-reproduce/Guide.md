你是我的“实验可视化 + 组会材料整理” agent。请基于我提供的 CoTra profiling 结果，生成可复现的 Python 作图代码和组会汇报提纲。

一、汇报目标
请不要把任务理解成“把表格画成图”，而要理解成“用图证明一个结论”：
当前这次 CoTra 复现（SIFT100M, 8 nodes x 8 search threads, ef=136, recall≈0.95）的主要瓶颈，更像是
1) task scheduling / queueing / coroutine overhead，
2) load imbalance / synchronization wait，
3) graph-access locality，
而不是 raw RDMA bandwidth 或 SIMD distance kernel。

二、输入数据
优先读取这些结构化文件；只有缺字段时，才回退到 markdown 文本：
- results/summary.tsv
- results/ann_profile_summary.tsv
- results/ann_node_summary.tsv
- results/network_perftest.tsv
- results/microbench_8t.tsv
- results/load_balance_node_metrics.tsv
- results/load_balance_summary.tsv
- Pasted markdown.md
- Pasted markdown (2).md

三、必须遵守的数据解释规则
1. rank0 打印的是 leader/global QPS，不是单节点 QPS。做 node-level balance 图时，使用 rank0_node_equiv_qps = leader_qps / 8。
2. latency 以 worker logs 为准，不要使用 rank0 的 0.00 latency。
3. profiler categories 不是严格 100% CPU-cycle decomposition；不要把所有 g_* 项硬堆成“总和=100%”的大饼图。
4. ann profiling run 的 QPS 是 breakdown/instrumented build 下的 QPS；请在图注里写清楚这不是 clean production QPS。
5. 优先用结构化 TSV 中的数值；markdown 里的结论只用于图注和 narrative，不要反过来硬编码数值。

四、输出文件
请生成以下交付物：
1. plot_cotra_breakdown.py
   - 可直接运行
   - 自动读取上述输入文件
   - 自动创建 figs/ 目录
   - 输出所有图的 PNG 和 PDF
2. figs/
   - 01_run_kpis.png/pdf
   - 02_profiler_major.png/pdf
   - 03_scheduler_detail.png/pdf
   - 04_network_headroom.png/pdf
   - 05_compute_gap.png/pdf
   - 06_balance_gap.png/pdf
   - 07_node_heatmap.png/pdf
3. meeting_outline.md
   - 按页组织组会内容
   - 每页包含：标题、这页要回答的问题、对应图片文件名、2~3句讲解词、1句结论
4. figure_notes.md
   - 每张图单独写：数据来源、构图方式、图想表达什么、不能过度解读什么

五、作图要求
1. 使用 Python + pandas + matplotlib。
2. 不要使用 seaborn。
3. 不要用饼图。
4. 优先使用：水平条形图、柱状图、折线图、热力图。
5. 风格统一，适合组会投影：
   - 背景简洁
   - 字体偏大
   - 线条和标签清楚
   - 每张图标题短，副标题写一句 takeaway
6. 每张图都必须带单位。
7. 对最关键的数值做 annotation。
8. 当两个量级相差超过 10x 时，可以用对数坐标，或改画成“占峰值百分比”。
9. 图中英文术语保留，如 QPS / RDMA / dist/s / wait / steal / post_stage。
10. 图标题和 markdown 讲稿用中文。

六、必须生成的图
图1：01_run_kpis
目标：给出这次 run 的 top-line 结果。
内容：
- recall
- QPS
- dist/query
- cluster-wide dist/s
形式：
- 4 个 KPI 卡片或一个简洁概览图
标题建议：
- “实验上下文与 top-line 结果”
图注必须写明：
- SIFT100M, 8 nodes x 8 threads, ef=136, Q=10000

图2：02_profiler_major
目标：回答“时间主要花在哪一层？”
数据：
- ann_profile_summary.tsv
内容：
- f_task_local
- f_wait_time
- f_task_remote
形式：
- 降序水平条形图
要求：
- 突出 f_wait_time 很高
- 图注注明：这是 profiling ratio，不是精确 CPU cycle decomposition

图3：03_scheduler_detail
目标：回答“为什么怀疑 orchestration / scheduler 是主瓶颈？”
数据：
- ann_profile_summary.tsv
内容：
- g_subquery_Q
- post_stage
- g_do_task
- post_task
- g_steal
- g_poll_proc
- g_query_Q
- g_result_Q
形式：
- 降序水平条形图
要求：
- 在图中高亮 g_subquery_Q、post_stage、g_steal
- 用较弱方式显示 g_poll_proc 很小

图4：04_network_headroom
目标：回答“网络是不是已经打满了？”
数据：
- in-run communication profile
- network_perftest.tsv
建议画法（二选一，优先更清楚的）：
A. 画“占峰值百分比”
   - in-run BW / 4096B perftest peak BW
   - in-run op rate / 128B perftest peak op rate
B. 或画原始值对比，并用 log scale
要求：
- 图上明确标注：
  - in-run ≈ 0.552 Gb/s/node
  - perftest ≈ 197 Gb/s
  - in-run ≈ 0.448 Mops/s/node
  - perftest ≈ 7.10 Mpps
- takeaway 必须是：raw bandwidth 不是当前一阶瓶颈

图5：05_compute_gap
目标：回答“compute kernel 本身是不是主瓶颈？”
数据：
- microbench_8t.tsv
- summary.tsv / markdown 中的 ANN dist/s
建议拆成同一张图中的两个区域，或者输出两个版本并选更好看的一个：
A. microbench kernel comparison
   - scalar no-vectorize
   - current uint8 L2
   - manual AVX2 L2
B. actual ANN vs extrapolated current-L2 capacity
   - actual ANN cluster dist/s
   - 8-node extrapolated current-L2 capacity
要求：
- 图中明确标注：
  - current vs scalar 的倍数
  - AVX2 vs current 只有很小增益
  - ANN 只用了 extrapolated microbench capacity 的很小一部分
- takeaway：SIMD 不是当前第一优先级，locality 更可疑

图6：06_balance_gap
目标：回答“系统看起来平衡，为什么还怀疑 imbalance？”
数据：
- load_balance_summary.tsv
内容：
- node-equivalent QPS CV
- worker latency CV
- f_wait_time CV
- g_subquery_Q CV
- g_steal CV
- post_stage CV
- read/write bandwidth CV（可选）
形式：
- 同一坐标系的柱状图
要求：
- 突出：QPS/latency CV 很小，但 g_steal 等内部指标 CV 很高
- takeaway：最终吞吐平衡，不代表内部执行健康

图7：07_node_heatmap
目标：回答“哪些节点的内部行为不一样？”
数据：
- load_balance_node_metrics.tsv
建议展示的列：
- f_wait_time
- g_subquery_Q
- g_steal
- post_stage
- read_MBps
- write_MBps
形式：
- heatmap，行为 node，列为 metric
要求：
- 标出 cpu1-5 和 cpu1-61 这两个比较典型的点
- 图注说明：这是 node-level symptom，不是 per-thread exact skew

七、meeting_outline.md 的组织方式
请按 8 页来组织，不要把所有图堆在一页：
第1页：问题与结论
第2页：实验设置与 top-line 结果
第3页：profiler 看到的第一阶瓶颈
第4页：为什么不是网络带宽瓶颈
第5页：为什么不是 SIMD/kernel 瓶颈
第6页：为什么“QPS 平衡”不等于“内部平衡”
第7页：当前最该优化的三个方向
第8页：下一步最值得做的补充实验

八、每页的讲稿要点
请给每页写 2~3 句口头讲解词，风格是“组会讲给系统同学听”，不要写成论文 prose。
口径要统一：
- 先回答问题
- 再给证据
- 最后给一句 take-away

九、最后一页的优化优先级
请明确给出当前建议的优化顺序：
1. scheduler / queueing / coroutine / post-stage 路径
2. wait / steal / synchronization / load balance
3. graph locality / metadata locality / cache/TLB behavior
4. network latency/control path
5. SIMD kernel
并说明：
- raw network bandwidth 暂时不是第一优先级
- 纯手写 AVX2 也暂时不是第一优先级

十、下一步实验建议
最后一页请加一个表格，列出：
- clean run without breakdown
- per-thread profiler rerun
- SEARCH_THREADS sweep
- perf stat on one rank
- ef sweep
每一项都要写：
- 目的
- 能区分什么假设
- 预期信号是什么

十一、代码质量要求
1. 代码模块化，至少包含：
   - load_data()
   - make_all_figures()
   - save_figure()
2. 对缺失文件给出友好报错。
3. 不要把数值硬编码到图里，尽量从文件中读取。
4. 图名、轴标签、注释文字写成常量，方便后续改。
5. 运行结束后，在终端打印：
   - 成功生成了哪些图
   - 每张图的输出路径
   - meeting_outline.md 的输出路径

请先输出：
1. 你理解的主线摘要（不超过 10 行）
2. 准备生成的图清单
3. 然后再给完整 Python 代码
4. 最后给 meeting_outline.md 内容