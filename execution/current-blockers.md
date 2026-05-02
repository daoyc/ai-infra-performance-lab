# 当前开放问题

本文件只保留当前最重要的 3 个问题。  
规则：

- 永远只保留 3 个最卡的问题
- 问题解决后，要么删除，要么标记为已解决
- 不记录泛泛问题，只记录阻塞当前实验推进的问题

## Question 1

- 问题：
  - 当前已经拿到第一组 `serve` baseline，下一步应优先如何设计 `output length` 单变量实验，才能区分输出长度对 `TTFT / TPOT / ITL / throughput` 的影响？
- 为什么卡住我：
  - 现在不再是“没有请求级指标”，而是已经有了 `TTFT / TPOT / ITL` baseline。如果下一步变量设计不干净，就很难判断变化到底来自输出长度、调度、随机输出长度，还是显存压力。
- 我准备怎么验证：
  - 固定 `num_prompts=60`、`input_len=128`、`request_rate=inf`、`max_concurrency=60`，只对比 `output_len=128 / 256 / 512`，每组至少记录 `TTFT / TPOT / ITL / output tok/s / request throughput / memory`。

## Question 2

- 问题：
  - 当前脚本通过 `3 * 20` 的请求和 `max_tokens=512` 明显在放大 `continuous batching` 与 `decode` 阶段。那么在现有 `vLLM + Qwen2-7B-Instruct + 4090 24G` 链路里，我该怎样把“当前观测更偏 decode”这件事，和 `prefill` 的影响边界区分开？
- 为什么卡住我：
  - 如果不能区分“这个脚本主要在测什么”和“它暂时没覆盖什么”，我就无法准确解释实验结果，更没法判断下一轮应该改 prompt 长度、输出长度还是并发。
- 我准备怎么验证：
  - 通过分别控制输入长度、输出长度和请求数量，做一轮单变量对比，观察哪些变化更明显影响首 token 延迟，哪些变化更明显影响生成吞吐。

## Question 3

- 问题：
  - 在当前脚本已经天然具备“60 请求 + 长输出”的前提下，下一步最值得优先拆开的变量顺序应该是什么：先改 `request_count / concurrency`，还是先改 `input/output length`？
- 为什么卡住我：
  - 当前脚本已经带有比较强的 workload 假设，如果我不先把变量拆开，就很难知道吞吐变化到底来自动态批处理、长输出，还是 prompt 本身。
- 我准备怎么验证：
  - 先固定模型、版本和硬件，只做单变量对比，并优先从最接近当前脚本意图的变量开始，把“现象 -> 指标变化 -> 假设”记入实验台账。

## 已解决问题

- `serve` benchmark 请求级指标补齐：
  - 已通过 `vllm bench serve` 跑通 `openai-chat + /v1/chat/completions`
  - 已拿到 3 轮同口径 `serve` baseline
  - 当前 AVG：`Mean TTFT=169.71 ms`、`Mean TPOT=21.28 ms`、`Mean ITL=20.36 ms`、`Output throughput=995.04 tok/s`
- `vim` 编辑中文乱码：
  - 临时方案：设置 `encoding / fileencoding / termencoding` 为 `utf-8`
  - 长期方案：在 `~/.vimrc` 中固化 `utf-8` 相关编码配置
