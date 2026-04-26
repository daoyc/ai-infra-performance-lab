# Inference Baseline

本文件作为统一实验台账，后续所有推理实验都沿用同一记录结构。

## 记录规则

- 先记录事实，再写解释
- 每组实验都要写“现象 -> 指标变化 -> 瓶颈假设”
- 不要一开始就扩展太多变量，优先控制：
  - `batch size`
  - `concurrency`
  - `input/output length`

## 实验表头

| Date | Stack | Model | Hardware | Workload | Batch | Concurrency | Input Length | Output Length | TTFT | TPOT | Throughput | Memory | Bottleneck Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待补充 | `vLLM` | 待补充 | `4090 24G` | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

## 已同步的历史进度

### 环境与版本

| Item | Value |
| --- | --- |
| CPU | `Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz` |
| Container Spec | `16U120G` |
| GPU | `NVIDIA GeForce RTX 4090 24G 450w` |
| OS | `Ubuntu 5.15.0-78-generic x86_64` |
| Model | `Qwen2-7B-Instruct` |
| vLLM | `0.10.1.1` |
| CUDA | `11.8` |

### 已验证的部署与校验链路

- 已完成云端 GPU 环境接入
- 已检查 `nvidia-smi / nvcc --version / python --version / vllm --version`
- 已通过 `huggingface-cli` 与镜像源配置模型下载链路
- 已通过 `transformers` 加载校验模型完整性

### 已同步的 benchmark 配置

| Field | Value |
| --- | --- |
| `model_name` | `./Qwen2-7B-Instruct` |
| `tensor_parallel_size` | `1` |
| `gpu_memory_utilization` | `0.88` |
| `dtype` | `auto (FP16/BF16)` |
| `enforce_eager` | `False (CUDA Graph)` |
| `temperature` | `0.7` |
| `base_prompts` | `3` |
| `prompts` | `3 * 20` |
| `request_count` | `60` |
| `max_tokens` | `512` |

### 已同步的 benchmark 脚本意图

- 当前 `benchmark.py` 不是泛化跑模型脚本，而是一个面向 `vLLM continuous batching` 的最小实验入口。
- 脚本重点是：
  - 触发 `scheduler` 对多请求进行动态批处理
  - 放大 `decode` 阶段，观察生成吞吐
  - 先拿到真实 `tokens/s` 与单次请求平均耗时
- 当前脚本的 workload 特征是：
  - 使用 3 条中文基准 prompt
  - 通过 `base_prompts * 20` 构造 60 个请求
  - 通过 `max_tokens=512` 拉长生成阶段，方便观察 `decode` 表现
- 当前脚本已直接输出：
  - 总生成 token 数
  - 总耗时
  - 平均吞吐 `tokens/s`
  - 单 prompt 平均耗时
- 当前脚本还没有直接拆出：
  - `TTFT`
  - `TPOT`
  - 显存占用
  - `prefill / decode` 的分阶段指标

### 已跑通的命令入口

#### 模型完整性校验

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
path = "./Qwen2-7B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
```

#### Benchmark 入口

```bash
python benchmark.py
```

#### Nsight Systems 采集入口

```bash
mkdir -p week1-test
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --sample=cpu \
  --cpuctxsw=trace \
  -d 30 \
  -y 60 \
  --output=./week1-test/week1_vllm_opt_batch16 \
  python benchmark.py \
  --block-sizes 16 \
  --batch-sizes 64
```

### 当前状态判断

- 当前不再是“从零开始准备环境”
- 当前已经进入“实验链路已打通，且具备 continuous batching 与 decode 吞吐观测入口，但指标语言与结果解释仍待补齐”的阶段
- 下一步重点不是继续折腾部署，而是把 benchmark 输出和 `TTFT / TPOT / throughput / memory` 建立稳定映射，并补齐分阶段观测手段

## 第一批实验任务

### Experiment 1：基线跑通

- 目标：
  - 跑通一条最小 `vLLM + GPU` 推理链路
  - 固化当前 `continuous batching + decode throughput` 基线
- 最少记录：
  - 模型
  - 请求方式
  - 请求数量
  - 总生成 tokens
  - 总耗时
  - throughput
  - 单 prompt 平均耗时
  - TTFT
  - TPOT
  - 显存占用
  - 当前瓶颈假设

### Experiment 2：Batch size 对比

- 目标：
  - 观察 batch size 变化对延迟、吞吐和显存的影响

### Experiment 3：Concurrency 对比

- 目标：
  - 观察并发变化对 TTFT、TPOT 和吞吐的影响

### Experiment 4：Input / Output Length 对比

- 目标：
  - 观察上下文长度和输出长度变化对推理表现的影响

## 结果解读模板

- 现象：
  - 待补充。
- 指标变化：
  - 待补充。
- 我的解释：
  - 待补充。
- 下一步验证：
  - 待补充。
