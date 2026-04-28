# 推理指标对照表

本文件用于把“当前已经跑出来的 benchmark 数字”和“它们在推理性能里到底代表什么”对齐。

## 当前判断

- 你现在已经能稳定跑出一条 `vLLM + GPU` 的 offline benchmark 基线。
- 当前这条基线更偏 `continuous batching + decode throughput` 观察。
- 当前最缺的不是新的部署动作，而是把 `TTFT / TPOT / memory / prefill / decode` 的指标语言真正建立起来。

## 旧计划与当前阶段对齐

`benchmark数据.xlsx` 里的旧计划可以保留为参考，但当前阶段需要调整：

- 已完成：
  - 云端 GPU 环境接入
  - `vLLM + Qwen2-7B-Instruct` 部署与模型完整性校验
  - baseline benchmark 跑通
  - `Nsight Systems` 入口准备
- 当前优先级上升：
  - 指标语言对齐
  - `TTFT / TPOT / memory` 补齐
  - `prefill / decode` 边界解释
- 当前优先级下降：
  - 继续折腾部署
  - 太早进入 `Nsight Compute` 或 kernel 细节

## 当前已同步的基线数据

来源：`benchmark数据.xlsx / 基线数据`

| Round | Total Generated Tokens | Total Time (s) | Output Throughput (tok/s) | Avg Per Prompt (s) |
| --- | --- | --- | --- | --- |
| 1 | `29374` | `11.21` | `2620.4` | `0.19` |
| 2 | `29374` | `10.85` | `2707.3` | `0.18` |
| 3 | `29374` | `10.85` | `2708.4` | `0.18` |
| AVG | `29374` | `10.97` | `2678.7` | `0.183` |

补充判断：

- `time(s)` 波动系数约为 `1.55%`
- `tokens/s` 波动系数约为 `1.54%`
- 这说明当前 baseline 的重复性已经够用，可以作为后续变量对比的起点

## 指标学习表

状态说明：

- `已遇到`：已经在当前 benchmark 或历史数据里直接出现
- `脚本已支持，待采集`：仓库里的脚本或路径已经支持，但你还没正式产出结果
- `后续阶段`：后面需要通过 serve benchmark、timeline 或 profiling 补齐

| 指标 | 当前状态 | 当前来源 | 它代表什么 | 更偏哪一阶段 | 当前该怎么理解 |
| --- | --- | --- | --- | --- | --- |
| `total_generated_tokens` | `已遇到` | `benchmark数据.xlsx` / offline benchmark | 本轮总共生成了多少输出 token | `decode` 更明显 | 它是总输出量，不是延迟，也不是单独的吞吐结论 |
| `total_time_s` | `已遇到` | `benchmark数据.xlsx` / offline benchmark | 整批 workload 从开始到结束的总墙钟时间 | `prefill + decode` 混合 | 适合做整体基线，但不能直接说明首 token 体验 |
| `output throughput (tok/s)` | `已遇到` | `benchmark数据.xlsx` / offline benchmark | 总输出 token / 总耗时，表示整体生成吞吐 | `decode` 更明显 | 这是你当前最可信的核心指标，但它不是完整用户体验指标 |
| `avg per prompt (s)` | `已遇到` | `benchmark数据.xlsx` / offline benchmark | 总耗时均摊到每个请求后的平均时间 | `prefill + decode` 混合 | 只能当粗粒度均值，不能当严格单请求 latency |
| `request_count` | `已遇到` | 当前 `benchmark.py` 配置 | 一轮 benchmark 发出了多少请求 | 调度层 | 你当前默认 workload 是 `3 * 20 = 60` 个请求 |
| `TTFT` | `脚本已支持，待采集` | `benchmarks/scripts/benchmark.py serve` | Time To First Token，首 token 延迟 | `prefill` 更明显 | 它更接近“用户第一次看到响应要等多久” |
| `TPOT` | `脚本已支持，待采集` | `benchmarks/scripts/benchmark.py serve` | Time Per Output Token，后续 token 平均生成时间 | `decode` 更明显 | 它更接近稳定生成阶段的速度，而不是首包速度 |
| `ITL` | `脚本已支持，待采集` | `vllm bench serve` | Inter-Token Latency，相邻输出 token 间隔 | `decode` | 通常和 TPOT 一起看，用来判断生成阶段是否平稳 |
| `E2E latency` | `脚本已支持，待采集` | `vllm bench serve` | 一个请求从进入到完整结束的总时延 | `prefill + decode + queue` | 更适合看完整请求体验 |
| `request throughput (req/s)` | `脚本已支持，待采集` | `vllm bench serve` | 每秒处理多少请求 | 调度层 | 它和 `tok/s` 不同，更像系统容量指标 |
| `GPU memory used / peak` | `脚本已支持，待采集` | 新版 `benchmarks/scripts/benchmark.py` | 运行过程中显存占用和峰值占用 | `KV cache / runtime` | 当前先把它当资源占用指标，不要急着把“显存高”直接解释为“性能好/差” |
| `gpu_cache_usage_perc` | `后续阶段` | vLLM metrics / Prometheus | KV cache 使用比例 | `KV cache` | 后面判断 cache 是否成为约束时会很重要 |
| `queue time` | `后续阶段` | vLLM metrics | 请求在真正执行前排队多久 | 调度层 | 并发提高后会影响 TTFT 与 E2E |
| `P50 / P95 / P99 latency` | `后续阶段` | `vllm bench serve` | 延迟分布，而不是平均值 | 调度层 | 平均值好看不代表尾延迟也好 |
| `GPU util / timeline busy` | `后续阶段` | `nsys profile` | GPU 到底忙不忙、有没有空洞 | runtime / 调度层 | 用来判断是不是 CPU-GPU 协作或 launch gap 问题 |
| `kernel hotspot` | `后续阶段` | Nsight / profiling | 时间主要花在哪些 kernel | kernel 层 | 后面才需要进这一步 |
| `occupancy / bandwidth` | `后续阶段` | Nsight Compute | 是算力墙还是访存墙 | kernel 层 | 这是更后面的分析语言，不是你当前第一优先级 |

## 当前最容易混淆的点

| 容易混淆的点 | 正确理解 |
| --- | --- |
| `tok/s` 高 = 用户体验一定好 | 不对。`tok/s` 更偏生成吞吐，`TTFT` 差的话，首包体验仍然可能很差 |
| `avg per prompt` = `TTFT` | 不对。`avg per prompt` 是总耗时均摊，不是首 token 延迟 |
| 当前 offline benchmark = 已经完整测到推理性能 | 不对。你现在更像是测到了 decode 吞吐基线，还没把请求级 latency 体系补齐 |
| 显存占用越高越差 | 不一定。显存被 KV cache 合理利用，可能反而有利于吞吐 |

## 接下来只需要补齐的东西

1. 用 `benchmarks/scripts/benchmark.py serve` 跑出第一份 `TTFT / TPOT / memory` 结果。
2. 把当前 offline benchmark 和 serve benchmark 的角色彻底分开：
   - offline：偏吞吐与 decode 基线
   - serve：偏请求级 latency 与 TTFT/TPOT
3. 做第一轮单变量对比时，优先顺序建议是：
   - `output length`
   - `request_count / concurrency`
   - `input length`
