# 学习推进闭环

本文件用于把当前这条 “从已有 CPU / 系统 / runtime 性能能力，迁移到 AI Infra / LLM 推理性能” 的学习过程做成可追踪闭环。

## 目的

不是泛泛学 AI，也不是先堆很多资料，而是围绕真实 benchmark 和真实指标，形成一条能复用的方法线：

- 我当前会什么
- 我缺什么
- 我用什么实验去补
- 我如何判断自己真的掌握了
- 我的学习方法哪里还能继续提效

## 当前阶段

你现在已经完成了：

- 云端 GPU 环境接入
- `vLLM + Qwen2-7B-Instruct` 部署与模型校验
- offline benchmark 基线跑通
- serve benchmark 基线跑通，并拿到第一组 `TTFT / TPOT / ITL`
- `Nsight Systems` 入口准备

你现在还需要补齐的是：

- `TTFT / TPOT / ITL` 随变量变化的解释
- `memory` 字段在每轮实验中的稳定记录
- `prefill / decode` 的边界解释和实验验证
- 从“会跑 benchmark”升级到“会解释 benchmark”

## 当前可迁移的旧能力

| 能力 | 为什么对现在有用 |
| --- | --- |
| CPU / runtime 性能分析习惯 | 你已经习惯先看指标、再解释，而不是只看结果 |
| 基线思维与重复性意识 | 你已经会做多轮 baseline，这对推理性能实验非常重要 |
| 系统级资源视角 | 你更容易理解资源占用、调度和吞吐之间的关系 |
| 报告与归因表达能力 | 后面形成可投递项目时，这部分会直接复用 |

## 当前必须补的新能力

| 新能力 | 当前缺口 | 如何补 |
| --- | --- | --- |
| `TTFT / TPOT / ITL / E2E` 指标语言 | 已有第一组 serve baseline，但还需要解释变量变化 | 用 output length 对比实验建立第一条因果解释 |
| `prefill / decode` 拆分视角 | 当前 offline benchmark 更偏 decode，边界还不够清楚 | 通过 `input/output length` 与 `request_count` 单变量实验拆开 |
| `memory / KV cache` 视角 | 当前 serve 表还缺少 memory 字段 | 把显存采样纳入 benchmark 台账 |
| `serve benchmark` 使用与解释 | 已能跑通，但还要形成解释模板 | 固定记录 `TTFT / TPOT / ITL / throughput / memory` |
| 推理实验台账化 | 已有数据，但还没完全变成可复盘闭环 | 每轮实验都写“现象 -> 指标变化 -> 解释 -> 下一步验证” |

## 固定学习闭环

每一轮学习都固定按下面 6 步走，不再临时切题：

1. 只问一个具体问题  
   例：`当前 batch 变大后，TTFT 为什么变差？`

2. 只选一组主指标  
   例：先只看 `TTFT + tok/s`，不要同时扩散到十几个指标

3. 只改一个变量  
   例：先只改 `output length`，不要同一轮同时改并发和输入长度

4. 先跑 3 轮基线，再跑对照组  
   例：先确认 baseline 稳定，再开始谈变化

5. 强制写解释  
   固定写法：
   - 现象
   - 指标变化
   - 我的解释
   - 下一步怎么验证

6. 回写仓库  
   至少同步到：
   - `../benchmarks/baselines.md`
   - `logs/`
   - 必要时 `../benchmarks/analysis/`

## 当前的追踪节奏

### Stage 1：补齐指标语言

- 跑通 `benchmarks/scripts/benchmark.py serve`
- 拿到第一组 `TTFT / TPOT / ITL`
- 用自己的话解释这些指标

状态：已完成 `TTFT / TPOT / ITL` 的第一组 baseline，`memory` 仍需在 serve 对比实验中补齐。

### Stage 2：建立阶段边界

- 用单变量实验区分 `prefill` 和 `decode`
- 先做：
  - `output length`
  - `request_count / concurrency`
  - `input length`

### Stage 3：形成第一版项目叙事

- 固定一组 baseline
- 固定一组对照实验
- 固定一份“指标 -> 归因 -> 结果”的小报告

## 当前最值得优化的学习方法

| 优化点 | 现在的问题 | 改进方向 |
| --- | --- | --- |
| 理论输入过宽 | 容易一口气学太多概念，和当前实验脱节 | 只补当前 benchmark 真正需要解释的概念 |
| 指标与实验没有严格绑定 | 容易看到数字，但不知道它支撑哪个判断 | 每次实验前先写“本轮主指标是什么” |
| 变量扩散太快 | 同时改太多参数，很难形成结论 | 严格单变量推进 |
| 只保存结果，不保存解释 | 后面容易忘记当时为什么这么判断 | 每轮都写最小解释卡片 |
| 方法没有复盘 | 学过一次但没沉淀成自己的套路 | 每周回头看：哪一步最费时、哪一步最模糊 |

## 每周固定复盘问题

每周只回答这 5 个问题：

1. 我这周只打通了哪一个主题？
2. 我这周新增了哪一个可复用指标或实验动作？
3. 我这周最重要的误解被纠正了什么？
4. 我这周的时间浪费在哪个环节？
5. 下周如果只保留一个重点，我该保留什么？

## 当前下一步

你接下来最应该做的不是继续看资料，而是：

1. 把第一组 `serve` baseline 作为固定参照。
2. 只做一轮单变量对比，先看 `output length = 128 / 256 / 512`。
3. 每轮补齐 `TTFT / TPOT / ITL / output tok/s / request throughput / memory`。
4. 写一条最小解释：
   - 当前 offline benchmark 主要在测什么
   - 当前 serve benchmark 补上了什么
   - output length 变化主要影响了哪些指标
