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

## 本周关键判断

- 当前已经不是“不会搭环境”的问题。
- 当前也不是“不会跑 benchmark”的问题。
- 当前真正卡住的是：
  - 不清楚 benchmark 输出和推理性能指标之间的严格映射
  - 还没把 `TTFT / TPOT / memory / prefill / decode` 建立成自己的分析语言

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

## 对技能升级的影响

- 本周新增的是“推理指标语言”和“benchmark 解释框架”的基础设施。
- 这仍然主要支撑 `LLM 推理性能` 方向的学习推进。
- 本周暂不升级 `identity/skills.md` 中的相关技能等级，因为还缺少一轮带 `TTFT / TPOT / memory` 的实测证据。

## 下周只做什么

1. 跑第一组 `serve` benchmark。
2. 拿到第一批 `TTFT / TPOT / memory` 数据。
3. 做一轮单变量实验，优先看 `output length`。
4. 写出第一条最小解释：
   - offline benchmark 主要测什么
   - serve benchmark 补上了什么
