# AI Infra Performance Lab

记录我从当前基础出发，向 `AI Infra / AI 性能工程 / LLM 推理优化` 方向迁移的过程。

## 主目录

### 1. `identity/`

- [目录入口](identity/README.md)
- [人物画像](identity/summary.md)
- [技能基线](identity/skills.md)

这一组只回答两件事：

- 我当前是谁
- 我已经验证了哪些能力

### 2. `execution/`

- [目录入口](execution/README.md)
- [路线图](execution/roadmap.md)
- [当前阻塞问题](execution/current-blockers.md)
- [概念与术语](execution/concepts.md)
- [学习方法](execution/method.md)
- [周推进记录](execution/weekly/week-001.md)
- [Week 002](execution/weekly/week-002.md)

这一组只回答三件事：

- 接下来做什么
- 当前卡在哪里
- 每周怎么推进

### 3. `benchmarks/`

- [目录入口](benchmarks/README.md)
- [基线台账](benchmarks/baselines.md)
- [benchmark 脚本](benchmarks/scripts/benchmark.py)
- [指标说明](benchmarks/metrics-guide.md)
- [实验分析](benchmarks/analysis/README.md)

这一组只回答两件事：

- 已经跑出了哪些数据
- 这些数据分别说明什么，还缺什么

### 4. 其他

- [Agent 规则](AGENTS.md)
- [工具目录](tools/README.md)

## 当前阶段

当前主线已经从“搭环境”推进到“解释 benchmark 数据”和“做单变量实验”：

- `vLLM + GPU` 环境与模型校验已经打通
- offline benchmark 基线已经跑通并具备重复性
- serve benchmark 已经跑通，并形成第一组 `TTFT / TPOT / ITL` 请求级 baseline
- 当前最关键的工作，是用 `output length` 单变量实验解释指标变化，并补齐每轮 memory 记录

## 记录规则

- `identity/` 放稳定结论，不堆过程
- `execution/` 放路线图、阻塞点、方法与周推进
- `benchmarks/` 放脚本、数据、指标解释与实验分析

后续新增内容优先落到这三条主线上，不再回到按 `docs / logs / reports` 分类的方式。
