#!/usr/bin/env python3
"""Benchmark entrypoints for this repository.

This file keeps two benchmark paths aligned with the current learning stage:

1. offline
   - Uses ``vllm.LLM.generate`` directly.
   - Good for continuous batching / decode throughput baselines.
   - Measures total generated tokens, total wall time, output throughput,
     average per-request time, and optional GPU memory sampling.
   - Does not directly emit TTFT / TPOT.

2. serve
   - Wraps the official ``vllm bench serve`` CLI.
   - Intended for a running ``vllm serve`` endpoint.
   - Good for collecting TTFT / TPOT / ITL / end-to-end latency style metrics.
   - Can save detailed benchmark results to JSON for later analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "./Qwen2-7B-Instruct"
DEFAULT_BASE_PROMPTS = [
    "请用中文详细介绍人工智能的发展历史和未来趋势。",
    "写一段关于 AI 性能优化的技术博客大纲。",
    "解释 Transformer 架构中的 Attention 机制及其在 GPU 上的优化点。",
]


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_prompts(multiplier: int) -> list[str]:
    return DEFAULT_BASE_PROMPTS * multiplier


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def pad_display(text: str, target_width: int) -> str:
    padding = max(target_width - display_width(text), 0)
    return text + (" " * padding)


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [[format_value(cell) for cell in row] for row in rows]
    widths = [display_width(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], display_width(cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in rendered_rows)
    return "\n".join(lines)


def print_section(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print(f"\n[{title}]")
    print(render_table(headers, rows))


def build_offline_judgments(result: dict[str, Any]) -> list[list[Any]]:
    config = result["config"]
    metrics = result["metrics"]
    memory = result["memory"]
    judgments: list[list[Any]] = []

    judgments.append(
        [
            "吞吐定位",
            "当前结果更适合作为 decode 吞吐基线",
            "offline 模式直接统计总输出 token / 总耗时，最能反映 continuous batching 下的整体生成效率。",
        ]
    )
    judgments.append(
        [
            "延迟边界",
            "当前不能判断首 token 体验",
            "TTFT / TPOT 为空，说明这轮还不能拿来解释首包慢不慢，也不能区分 prefill 和 decode 的分阶段延迟。",
        ]
    )

    if memory.get("peak_usage_pct") is not None:
        peak_pct = float(memory["peak_usage_pct"])
        if peak_pct >= 90:
            summary = "显存压力已经偏高"
            detail = "峰值显存占比已经接近满载，后续如果提高并发、输出长度或上下文长度，要优先关注 KV cache 挤压与 OOM 风险。"
        elif peak_pct >= 80:
            summary = "显存占用较高但仍可继续观察"
            detail = "当前已经进入高显存区间，适合继续做小步参数实验，但不宜一次性把多个变量同时拉高。"
        else:
            summary = "显存仍有一定余量"
            detail = "当前更适合继续做 batch、并发或输出长度对比，先观察吞吐与显存变化关系。"
        judgments.append(["显存判断", summary, detail])

    request_count = config.get("request_count")
    if request_count and int(request_count) > 1:
        judgments.append(
            [
                "工作负载形态",
                "当前结果含有明显批处理效应",
                f"这轮共有 {request_count} 个请求，vLLM scheduler 会主动做动态批处理，所以结果不等价于单请求纯延迟。",
            ]
        )

    temperature = config.get("temperature")
    if temperature is not None and float(temperature) > 0:
        judgments.append(
            [
                "可重复性提醒",
                "当前总 token 数可能存在轻微波动",
                "temperature 大于 0 会带来采样随机性；如果后面要做严格性能校准，建议再补一轮 temperature=0 的基线。",
            ]
        )

    avg_req = metrics.get("avg_request_time_s")
    if avg_req is not None:
        judgments.append(
            [
                "均摊单请求耗时",
                "只能做粗粒度均值参考",
                "这个值是总耗时除以请求数，不是请求级 P50/P95 latency，也不是 TTFT；后续不要把它直接当成真实用户体验指标。",
            ]
        )

    return judgments


def build_serve_judgments(summary: dict[str, Any]) -> list[list[Any]]:
    interesting = summary.get("interesting_metrics", {})
    judgments: list[list[Any]] = []

    has_ttft = any("ttft" in key.lower() for key in interesting)
    has_tpot = any("tpot" in key.lower() or "itl" in key.lower() for key in interesting)

    if has_ttft:
        judgments.append(
            [
                "首包视角",
                "这轮已经可以开始看首 token 体验",
                "只要结果里出现 TTFT，就可以开始讨论 prefill、排队和首包响应时间。",
            ]
        )
    else:
        judgments.append(
            [
                "首包视角",
                "这轮还没拿到可靠 TTFT",
                "如果结果文件里没有 TTFT 字段，就还不能完整回答首 token 为什么快或慢。",
            ]
        )

    if has_tpot:
        judgments.append(
            [
                "生成阶段视角",
                "这轮已经可以开始看稳定生成速度",
                "TPOT / ITL 更适合解释 decode 阶段的生成节奏，可以和 offline 的 tok/s 形成互补。",
            ]
        )
    else:
        judgments.append(
            [
                "生成阶段视角",
                "这轮还没完整拿到 token 级延迟",
                "如果没有 TPOT / ITL，就还不能把请求级延迟和稳定生成速度拆开解释。",
            ]
        )

    memory = summary.get("memory", {})
    peak_pct = memory.get("peak_usage_pct")
    if peak_pct is not None:
        peak_pct = float(peak_pct)
        if peak_pct >= 90:
            judgments.append(
                [
                    "显存判断",
                    "显存占用接近满载",
                    "后续如果 TTFT、TPOT 变差，同时显存峰值继续上升，就要优先怀疑 KV cache 与并发压力。",
                ]
            )
        elif peak_pct >= 80:
            judgments.append(
                [
                    "显存判断",
                    "显存占用较高",
                    "适合继续做 request_count / output_len 对比，观察吞吐和 latency 是否一起恶化。",
                ]
            )

    return judgments


@dataclass
class MemorySummary:
    gpu_index: int
    available: bool
    total_mb: int | None = None
    used_start_mb: int | None = None
    used_end_mb: int | None = None
    peak_used_mb: int | None = None
    peak_usage_pct: float | None = None
    note: str | None = None


class NvidiaMemoryMonitor:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[int, int]] = []
        self._note: str | None = None

    @staticmethod
    def _has_nvidia_smi() -> bool:
        return shutil.which("nvidia-smi") is not None

    def _query(self) -> tuple[int, int] | None:
        if not self._has_nvidia_smi():
            self._note = "nvidia-smi 不可用，未采集显存数据。"
            return None
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self._note = f"显存采集失败：{exc}"
            return None

        line = proc.stdout.strip().splitlines()[0]
        used_str, total_str = [part.strip() for part in line.split(",")]
        return int(used_str), int(total_str)

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            sample = self._query()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        initial = self._query()
        if initial is not None:
            self._samples.append(initial)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> MemorySummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 5))
        final = self._query()
        if final is not None:
            self._samples.append(final)

        if not self._samples:
            return MemorySummary(
                gpu_index=self.gpu_index,
                available=False,
                note=self._note or "没有采集到显存样本。",
            )

        used_values = [used for used, _ in self._samples]
        total_mb = self._samples[-1][1]
        peak_used_mb = max(used_values)
        return MemorySummary(
            gpu_index=self.gpu_index,
            available=True,
            total_mb=total_mb,
            used_start_mb=self._samples[0][0],
            used_end_mb=self._samples[-1][0],
            peak_used_mb=peak_used_mb,
            peak_usage_pct=round(peak_used_mb / total_mb * 100, 2) if total_mb else None,
            note=self._note,
        )


def save_json(payload: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_print_json(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.print_json:
        print("\n[raw-json]")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_offline_summary(result: dict[str, Any]) -> None:
    config = result["config"]
    metrics = result["metrics"]
    memory = result["memory"]
    coverage = result["coverage"]

    print("\n=== Offline 基线测试摘要 ===")
    print("说明：这组结果更适合观察 continuous batching 下的 decode 吞吐，不直接提供 TTFT / TPOT。")

    print_section(
        "测试工作负载",
        ["字段", "取值"],
        [
            ["测试模式", result["mode"]],
            ["模型路径", config["model_name"]],
            ["请求总数", config["request_count"]],
            ["prompt 重复倍数", config["prompt_multiplier"]],
            ["最大输出 token", config["max_tokens"]],
            ["temperature", config["temperature"]],
        ],
    )
    print_section(
        "运行配置",
        ["字段", "取值"],
        [
            ["张量并行数", config["tensor_parallel_size"]],
            ["显存利用率上限", config["gpu_memory_utilization"]],
            ["dtype", config["dtype"]],
            ["enforce_eager", config["enforce_eager"]],
        ],
    )
    print_section(
        "核心指标",
        ["指标", "数值", "解释"],
        [
            ["总输出 token", metrics["total_generated_tokens"], "这一轮总共生成了多少 token"],
            ["总耗时(秒)", metrics["total_time_s"], "60 个请求从开始到结束的总耗时"],
            ["输出吞吐(tok/s)", metrics["output_token_throughput_tps"], "整体生成吞吐，当前更偏 decode"],
            ["均摊单请求耗时(秒)", metrics["avg_request_time_s"], "总耗时均摊到每个请求后的结果"],
            ["TTFT(ms)", metrics["ttft_ms"], "offline 模式不直接提供"],
            ["TPOT(ms)", metrics["tpot_ms"], "offline 模式不直接提供"],
        ],
    )
    print_section(
        "显存观测",
        ["字段", "取值"],
        [
            ["GPU 编号", memory["gpu_index"]],
            ["是否采集成功", memory["available"]],
            ["总显存(MB)", memory["total_mb"]],
            ["起始显存(MB)", memory["used_start_mb"]],
            ["结束显存(MB)", memory["used_end_mb"]],
            ["显存峰值(MB)", memory["peak_used_mb"]],
            ["峰值占比(%)", memory["peak_usage_pct"]],
            ["备注", memory["note"]],
        ],
    )
    print_section(
        "当前覆盖范围",
        ["类别", "内容"],
        [
            ["本次已测到", coverage["measured_now"]],
            ["offline 还不能直接测到", coverage["not_directly_measured_in_offline_mode"]],
        ],
    )
    print_section(
        "结果判断",
        ["维度", "结论", "判断说明"],
        build_offline_judgments(result),
    )


def print_serve_summary(summary: dict[str, Any]) -> None:
    config = summary["config"]
    memory = summary["memory"]
    interesting = summary.get("interesting_metrics", {})

    print("\n=== Serve 基线测试摘要 ===")
    print("说明：这组结果用于补齐请求级 latency 指标，例如 TTFT / TPOT / ITL / E2E latency。")

    print_section(
        "请求配置",
        ["字段", "取值"],
        [
            ["后端", config["backend"]],
            ["base_url", config["base_url"]],
            ["endpoint", config["endpoint"]],
            ["模型名", config["model"]],
            ["dataset_name", config["dataset_name"]],
            ["请求数", config["num_prompts"]],
            ["输入长度", config["input_len"]],
            ["输出长度", config["output_len"]],
            ["request_rate", config["request_rate"]],
            ["最大并发", config["max_concurrency"]],
            ["本轮总耗时(秒)", summary["run_time_s"]],
            ["结果文件", summary["result_json"]],
        ],
    )
    print_section(
        "显存观测",
        ["字段", "取值"],
        [
            ["GPU 编号", memory["gpu_index"]],
            ["是否采集成功", memory["available"]],
            ["总显存(MB)", memory["total_mb"]],
            ["起始显存(MB)", memory["used_start_mb"]],
            ["结束显存(MB)", memory["used_end_mb"]],
            ["显存峰值(MB)", memory["peak_used_mb"]],
            ["峰值占比(%)", memory["peak_usage_pct"]],
            ["备注", memory["note"]],
        ],
    )

    if interesting:
        interesting_rows = [[key, value] for key, value in sorted(interesting.items())]
        print_section("解析出的关键指标", ["指标", "数值"], interesting_rows)
    else:
        print_section(
            "解析出的关键指标",
            ["指标", "数值"],
            [["备注", summary.get("note", "未解析到指标，请检查 vllm bench serve 输出。")]],
        )
    print_section(
        "结果判断",
        ["维度", "结论", "判断说明"],
        build_serve_judgments(summary),
    )


def run_offline(args: argparse.Namespace) -> int:
    try:
        import torch.cuda.nvtx as nvtx
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        print(
            "offline 模式需要当前环境安装 vLLM 与 torch。"
            f" 导入失败：{exc}",
            file=sys.stderr,
        )
        return 1

    prompts = build_prompts(args.prompt_multiplier)
    monitor = NvidiaMemoryMonitor(gpu_index=args.gpu_index, interval_s=args.memory_poll_interval_s)
    monitor.start()

    print("正在初始化 vLLM，这一步会准备 KV cache 与 scheduler ...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.nvtx:
        nvtx.range_push("repo_offline_generate")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    if args.nvtx:
        nvtx.range_pop()

    memory = monitor.stop()
    total_time_s = end_time - start_time
    total_generated_tokens = sum(
        len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs
    )

    result = {
        "mode": "offline",
        "description": (
            "direct vLLM.generate baseline; strongest for continuous batching / "
            "decode throughput observation, not direct TTFT/TPOT measurement"
        ),
        "config": {
            "model_name": args.model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_multiplier": args.prompt_multiplier,
            "request_count": len(prompts),
        },
        "metrics": {
            "total_generated_tokens": total_generated_tokens,
            "total_time_s": round(total_time_s, 4),
            "output_token_throughput_tps": round(
                total_generated_tokens / total_time_s, 4
            ) if total_time_s else None,
            "avg_request_time_s": round(total_time_s / len(prompts), 4) if prompts else None,
            "ttft_ms": None,
            "tpot_ms": None,
        },
        "memory": asdict(memory),
        "coverage": {
            "measured_now": [
                "total_generated_tokens",
                "total_time_s",
                "output_token_throughput_tps",
                "avg_request_time_s",
                "gpu_memory_peak_mb",
            ],
            "not_directly_measured_in_offline_mode": [
                "TTFT",
                "TPOT",
                "ITL",
                "P50/P95/P99 latency",
            ],
        },
    }

    print_offline_summary(result)
    maybe_print_json(result, args)
    save_json(result, args.output_json)
    del llm
    time.sleep(1)
    return 0


def add_metadata_args(cmd: list[str], metadata: list[str]) -> None:
    for item in metadata:
        cmd.extend(["--metadata", item])


def add_dataset_length_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.dataset_name == "random":
        cmd.extend(["--random-input-len", str(args.input_len)])
        cmd.extend(["--random-output-len", str(args.output_len)])
        return
    cmd.extend(["--input-len", str(args.input_len)])
    cmd.extend(["--output-len", str(args.output_len)])


def resolve_endpoint_type(args: argparse.Namespace) -> str:
    if args.endpoint_type:
        return args.endpoint_type
    if "chat/completions" in args.endpoint:
        return "openai-chat"
    return "openai"


def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_metrics(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            flat.update(flatten_metrics(value, next_prefix))
    else:
        flat[prefix] = obj
    return flat


def extract_interesting_metrics(payload: Any) -> dict[str, Any]:
    interesting_terms = (
        "ttft",
        "tpot",
        "itl",
        "latency",
        "throughput",
        "concurrency",
        "request_rate",
        "token",
        "e2el",
    )
    flat = flatten_metrics(payload)
    return {
        key: value
        for key, value in flat.items()
        if any(term in key.lower() for term in interesting_terms)
    }


def run_serve(args: argparse.Namespace) -> int:
    if shutil.which("vllm") is None:
        print("serve 模式需要当前环境能直接执行 `vllm` CLI。", file=sys.stderr)
        return 1

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / args.result_filename
    endpoint_type = resolve_endpoint_type(args)

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--endpoint-type",
        endpoint_type,
        "--base-url",
        args.base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        args.result_filename,
    ]
    add_dataset_length_args(cmd, args)

    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if args.disable_tqdm:
        cmd.append("--disable-tqdm")
    if args.no_stream:
        cmd.append("--no-stream")
    add_metadata_args(cmd, args.metadata)

    monitor = NvidiaMemoryMonitor(gpu_index=args.gpu_index, interval_s=args.memory_poll_interval_s)
    monitor.start()
    start_time = time.perf_counter()
    proc = subprocess.run(cmd, text=True)
    end_time = time.perf_counter()
    memory = monitor.stop()

    summary: dict[str, Any] = {
        "mode": "serve",
        "description": (
            "wrap official vllm bench serve; use this mode when TTFT / TPOT / ITL / "
            "end-to-end latency are the target metrics"
        ),
        "return_code": proc.returncode,
        "config": {
            "backend": args.backend,
            "endpoint_type": endpoint_type,
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "dataset_name": args.dataset_name,
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
            "command": cmd,
        },
        "run_time_s": round(end_time - start_time, 4),
        "memory": asdict(memory),
        "result_json": str(result_path),
    }

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary["interesting_metrics"] = extract_interesting_metrics(payload)
    else:
        summary["interesting_metrics"] = {}
        if proc.returncode != 0:
            summary["note"] = (
                "vllm bench serve 执行失败，未生成结果文件。"
                " 请优先检查命令参数与服务端模型是否匹配。"
            )
        else:
            summary["note"] = "未找到保存结果文件，请确认 vllm bench serve 已成功输出结果。"

    print_serve_summary(summary)
    maybe_print_json(summary, args)
    save_json(summary, args.summary_json)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repository benchmark helper for offline throughput and serve latency metrics."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    offline = subparsers.add_parser(
        "offline",
        help="Use direct vLLM.generate for throughput-oriented baselines.",
    )
    offline.add_argument("--model-name", default=DEFAULT_MODEL_PATH)
    offline.add_argument("--tensor-parallel-size", type=int, default=1)
    offline.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    offline.add_argument("--dtype", default="auto")
    offline.add_argument("--enforce-eager", type=str2bool, default=False)
    offline.add_argument("--temperature", type=float, default=0.7)
    offline.add_argument("--max-tokens", type=int, default=512)
    offline.add_argument("--prompt-multiplier", type=int, default=20)
    offline.add_argument("--gpu-index", type=int, default=0)
    offline.add_argument("--memory-poll-interval-s", type=float, default=0.1)
    offline.add_argument("--nvtx", action="store_true")
    offline.add_argument("--output-json")
    offline.add_argument("--print-json", action="store_true")

    serve = subparsers.add_parser(
        "serve",
        help="Wrap official `vllm bench serve` for TTFT / TPOT style metrics.",
    )
    serve.add_argument("--backend", default="openai")
    serve.add_argument("--base-url", default="http://127.0.0.1:8000")
    serve.add_argument("--endpoint-type", choices=["openai", "openai-chat"], default=None)
    serve.add_argument("--endpoint", default="/v1/chat/completions")
    serve.add_argument("--model", default="./Qwen2-7B-Instruct")
    serve.add_argument("--tokenizer")
    serve.add_argument("--dataset-name", default="random")
    serve.add_argument("--num-prompts", type=int, default=60)
    serve.add_argument("--input-len", type=int, default=128)
    serve.add_argument("--output-len", type=int, default=512)
    serve.add_argument("--request-rate", default="inf")
    serve.add_argument("--max-concurrency", type=int, default=60)
    serve.add_argument("--result-dir", default="benchmarks/results")
    serve.add_argument("--result-filename", default="serve-benchmark.json")
    serve.add_argument("--summary-json")
    serve.add_argument("--metadata", nargs="*", default=[])
    serve.add_argument("--gpu-index", type=int, default=0)
    serve.add_argument("--memory-poll-interval-s", type=float, default=0.1)
    serve.add_argument("--disable-tqdm", action="store_true")
    serve.add_argument("--no-stream", action="store_true")
    serve.add_argument("--print-json", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "offline":
        return run_offline(args)
    if args.mode == "serve":
        return run_serve(args)
    parser.error(f"Unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
