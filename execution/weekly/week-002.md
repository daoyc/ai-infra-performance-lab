# Week 002

## 本周目标

把“会跑 benchmark”推进到“知道这些 benchmark 数字在说明什么”，并把已有学习过程变成可追踪闭环。

## 本周输入

- `benchmark数据.xlsx`
- 外部 `benchmark.py`
- 已有仓库路线图、基线台账与 `week-001-vllm-setup` 报告

## 本周完成

1. 把历史 baseline 数据同步进仓库，确认当前 offline benchmark 已具备可复现性。
2. 把外部 `benchmark.py` 收进仓库，并明确它当前的角色是：
   - 偏 `continuous batching + decode throughput` 的 offline 基线脚本
   - 不是完整的 `TTFT / TPOT` 请求级 latency benchmark
3. 给 benchmark 增加下一版路径：
   - `offline` 模式保留吞吐基线
   - `serve` 模式对接官方 `vllm bench serve`
   - 增加 GPU 显存采样入口
4. 新增推理指标对照表与学习闭环文档，便于后续方法论迭代。
5. 通过 `serve` benchmark 跑通第一组请求级 latency baseline，并记录 3 轮同口径数据。

## 本周关键判断

- 当前已经不是“不会搭环境”的问题。
- 当前也不是“不会跑 benchmark”的问题。
- 当前已经完成从“只会跑 benchmark”到“能拿到 offline + serve 两类基线”的跃迁。
- 当前真正卡住的是：
  - 如何解释 `TTFT / TPOT / ITL` 随变量变化的原因
  - 如何把 `memory / KV cache` 纳入每轮对比记录
  - 如何用单变量实验拆开 `prefill / decode` 的影响边界

## 本周已确认的事实

- 当前 baseline 数据中，`time(s)` 与 `tokens/s` 的波动已经比较小，可以继续作为后续实验基线。
- 当前 offline benchmark 最可信的输出是：
  - 总生成 token 数
  - 总耗时
  - 输出吞吐 `tok/s`
  - 平均每请求耗时
- 当前 offline benchmark 还不能直接替代：
  - `TTFT`
  - `TPOT`
  - `ITL`
  - 请求级延迟分布
- 当前 serve benchmark 已经补齐：
  - `Mean TTFT = 169.71 ms`
  - `Mean TPOT = 21.28 ms`
  - `Mean ITL = 20.36 ms`
  - `Output throughput = 995.04 tok/s`
  - `Request throughput = 6.01 req/s`

## 对技能升级的影响

- 本周新增的是“推理指标语言”和“benchmark 解释框架”的基础设施。
- 这仍然主要支撑 `LLM 推理性能` 方向的学习推进。
- 本周可以认为 `LLM 推理性能指标理解` 从“发展中”推进到“部分验证”：已经有实测 `serve` baseline，但还缺少单变量对比与稳定解释。
- 暂不升级为“能实践”，因为还缺少 `output length / concurrency / input length` 至少一轮对比实验。

## 下周只做什么

1. 做一轮 `output length` 单变量实验，优先看 `128 / 256 / 512`。
2. 每轮补齐 `TTFT / TPOT / ITL / output tok/s / request throughput / memory`。
3. 写出第一条最小解释：
   - offline benchmark 主要测什么
   - serve benchmark 补上了什么
   - output length 变化主要影响了哪些指标
