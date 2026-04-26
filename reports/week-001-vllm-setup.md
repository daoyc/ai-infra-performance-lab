# Week 001 - vLLM Setup

## 摘要

本周的重点不是深入推理性能分析本身，而是把 `vLLM + GPU` 的实验入口先打通。  
当前已经完成了云端 `4090 24G` 环境准备、`Qwen2-7B-Instruct` 模型下载与完整性校验、`vLLM` 基础 benchmark 入口验证，以及 `Nsight Systems` 采集入口准备。

这意味着当前阶段已经不再是“从零开始研究怎么部署”，而是进入了“实验链路已打通，接下来需要把指标语言和结果解释补齐”的状态。

## 已完成进度

### 1. 云端 GPU 环境接入

- 使用 `AutoDL` 作为云端算力入口。
- 成功接入远程服务器并完成基础环境检查。
- 已验证以下基础命令可用：
  - `nvidia-smi`
  - `nvcc --version`
  - `python --version`
  - `vllm --version`

### 2. 模型下载与完整性校验

- 目标模型：`Qwen2-7B-Instruct`
- 已配置 `HF_ENDPOINT=https://hf-mirror.com` 以适配国内下载环境。
- 已尝试两类下载方式：
  - 默认单线程下载
  - 借助 `hf-transfer` / `snapshot_download` 的并发下载
- 已通过 `transformers` 完成模型文件完整性校验，说明当前模型目录可以被正常加载。

模型完整性校验入口：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "./Qwen2-7B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
```

### 3. vLLM benchmark 入口打通

- 已具备基础 benchmark 脚本入口：

```bash
python benchmark.py
```

- 当前已记录的一组基础配置：
  - `Model`: `Qwen2-7B-Instruct`
  - `vLLM`: `0.10.1.1`
  - `CUDA`: `11.8`
  - `GPU`: `NVIDIA GeForce RTX 4090 24G`
  - `tensor_parallel_size`: `1`
  - `gpu_memory_utilization`: `0.88`
  - `dtype`: `auto (FP16/BF16)`
  - `enforce_eager`: `False (CUDA Graph)`
  - `prompts`: `3 * 20`
  - `max_tokens`: `512`

### 4. Nsight Systems 采集入口准备

- 已完成 `nsight-systems` 安装与路径确认思路。
- 已准备一条可复用的采集命令模板：

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

当前价值不在于“已经做了深 profiling”，而在于：后续只要实验链路稳定，就能很快进入 trace 采集阶段。

## 当前实验环境

| Item | Value |
| --- | --- |
| CPU | `Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz` |
| Container Spec | `16U120G` |
| GPU | `NVIDIA GeForce RTX 4090 24G 450w` |
| OS | `Ubuntu 5.15.0-78-generic x86_64` |
| Model | `Qwen2-7B-Instruct` |
| vLLM | `0.10.1.1` |
| CUDA | `11.8` |

## 当前判断

当前已经完成的是：

- 部署链路
- 模型链路
- benchmark 入口
- profiling 入口

当前还没有真正补齐的是：

- `TTFT / TPOT / throughput / memory` 的指标语言
- `prefill / decode / KV cache` 和实际实验现象之间的对应关系
- benchmark 输出如何转成“现象 -> 指标变化 -> 性能解释”

换句话说，当前不是“不会搭环境”，而是“已经能开始做推理性能分析，但还没把分析语言建立起来”。

## 本周遇到的问题

### vim 编辑中文乱码

临时方案：

```vim
:set encoding=utf-8
:set fileencoding=utf-8
:set termencoding=utf-8
```

长期方案：

```vim
set encoding=utf-8
set fileencodings=utf-8,gbk,latin1
set fileencoding=utf-8
set termencoding=utf-8
```

## 下一步

接下来的优先级不再是继续折腾部署，而是：

1. 用自己的语言补齐 `decoder-only / prefill / decode / KV cache / TTFT / TPOT`
2. 把 benchmark 输出字段和这些指标建立明确映射
3. 先围绕 3 类变量做最小实验对比：
   - `batch size`
   - `concurrency`
   - `input/output length`
4. 在实验台账中固定记录：
   - 现象
   - 指标变化
   - 瓶颈假设

## 结论

这一周最重要的成果不是“已经得到了推理性能结论”，而是**已经把后续进入推理性能分析所需的基础实验入口打通**。  
从迁移路径上看，这一步的意义在于：你已经从“计划学 vLLM”进入了“可以围绕真实实验边干边学 vLLM”的阶段。
