# Benchmarks

这一组文件集中记录推理实验脚本、数据基线、指标解释和实验分析。

## 文件说明

- [基线台账](baselines.md)
  - 统一记录 baseline、变量对比和结果解释
- [benchmark 脚本](scripts/benchmark.py)
  - 当前仓库内的正式脚本入口
  - 已拆成 `offline / serve` 两条路径
- [指标说明](metrics-guide.md)
  - 解释当前已遇到和未遇到的推理性能指标
- [实验分析](analysis/README.md)
  - 收纳阶段报告、实验总结与复盘材料

## 当前状态

- 已完成云端 GPU 环境准备
- 已验证 `Qwen2-7B-Instruct` 可下载并完成完整性校验
- 已有 offline benchmark 基线
- 已有 `vllm bench serve` 方向的脚本入口
- 已同步 3 轮 offline baseline 数据

## 这组文件回答什么

- 我已经跑出了哪些数字
- 这些数字代表什么
- 下一轮实验该怎么跑
