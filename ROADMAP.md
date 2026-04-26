# AI Infra / 推理性能迁移路线图

## 总体方向

迁移主线保持不变：保留当前 `CPU / 系统 / 分布式性能优化` 身份，直接向 `AI Infra / LLM 推理性能` 岗位迁移。  
当前阶段不把 `CUDA kernel` 深挖作为起步目标，也不把通用 AI 应用开发作为主路线。

## 当前进度

根据已有学习记录，当前已经完成或跑通的内容包括：

- 已在云端 GPU 环境上完成基础实验环境搭建
- 已验证 `4090 24G + vLLM 0.10.1.1 + CUDA 11.8 + Qwen2-7B-Instruct` 这一组合可用
- 已完成模型下载、完整性校验、benchmark 入口和 Nsight Systems 采集入口的打通
- 已记录一条基础 benchmark 配置链路，说明实际进度已经提前进入 `Week 2` 的实验准备阶段

这意味着当前路线不是从零开始，而是：

- `Week 1` 的“推理链路和指标语言”仍需系统补齐
- `Week 2` 的 `vLLM + GPU` 实验入口实际上已经提前跑通

## Phase 1：Week 1

目标：一次性打通“推理链路和指标语言”。

本周只做 4 件事：

1. 用自己的话解释 6 个核心词：
   - `decoder-only`
   - `prefill`
   - `decode`
   - `KV cache`
   - `TTFT`
   - `TPOT`
2. 把这 6 个词和你现有的 `CPU / runtime / 系统性能` 经验建立对应关系。
3. 写出一条最小推理链路说明：
   - 请求进入模型
   - prefill 发生什么
   - decode 发生什么
   - KV cache 为什么关键
4. 整理当前最卡的 3 个问题，不扩散问题池。

本周产物：

- `docs/llm-inference-lexicon.md`
- `docs/open-questions.md`
- 一版可口头复述的推理链路说明
- 已有部署与 benchmark 进度同步进实验台账

本周验收：

- 能完整解释 `decoder-only / prefill / decode / KV cache / TTFT / TPOT`
- 能说明 TTFT、TPOT、throughput、显存占用分别反映什么
- 能讲清“为什么推理性能的观察不能只看吞吐”

## Phase 2：Week 2-4

目标：直接进入 `vLLM + GPU` 单卡实验，边干边学。

默认实验环境：

- 云端单卡 GPU
- 优先 `4090 24G`

执行顺序：

1. 跑通最小实验链路：
   - 模型启动
   - 请求发送
   - 指标记录
   - 结果归档
2. 只围绕 3 类变量做第一批实验：
   - `batch size`
   - `concurrency`
   - `input/output length`
3. 所有实验统一记录：
   - `TTFT`
   - `TPOT`
   - `throughput`
   - `显存占用`
   - `当前瓶颈假设`

学习规则：

- 先跑实验，再学对应问题
- 不做无问题驱动的资料堆积
- 不做无问题驱动的 `vLLM` 全源码通读

阶段产物：

- `benchmarks/inference-baseline.md`
- 至少 1 组基线记录
- 至少 3 组对比实验
- 已同步现有环境、版本与 benchmark 入口配置

阶段验收：

- Week 2：已跑通一条 `vLLM + GPU` 最小实验链路
- Week 2：已形成第一版瓶颈假设
- Week 4：每组实验都有“现象 -> 指标变化 -> 解释”
- Week 4：能明确讲出 prefill 和 decode 的性能差异

## Phase 3：Week 5-8

目标：开始做 runtime 观察、岗位语言沉淀和公开证据化。

本阶段聚焦：

- `continuous batching`
- `scheduler` 行为
- `KV cache` 管理
- 吞吐、延迟、显存之间的权衡

分析顺序固定为：

1. 应用层指标
2. runtime 日志 / trace
3. 最后才是 `Nsight Systems`

目标不是立刻深入 CUDA kernel，而是建立：

- `runtime 设计 -> 指标变化 -> 性能结果`

阶段产物：

- 一份推理链路与指标语言总结
- 一份单卡 `vLLM` 实验结果总结
- 一份 runtime 观察与瓶颈假设总结

阶段验收：

- 已形成可公开展示的实验报告
- 能围绕这些内容讲一版面试式迁移叙事
- `AI Infra / 推理性能` 不再只是方向，而开始有岗位级证据支撑
