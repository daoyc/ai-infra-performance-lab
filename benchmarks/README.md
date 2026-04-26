# Benchmarks

记录推理实验设计、指标结果、瓶颈假设与后续对比结论。

当前默认实验主线：

- `vLLM + GPU`
- 云端单卡
- 优先 `4090 24G`

当前已同步进来的历史进度：

- 已完成云端 GPU 环境准备
- 已验证 `Qwen2-7B-Instruct` 可下载并完成完整性校验
- 已有基础 benchmark 入口
- 已有 Nsight Systems 采集入口

实验记录统一入口：

- [推理基线台账](inference-baseline.md)
