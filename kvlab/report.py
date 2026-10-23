"""仅从真实 JSONL 记录生成静态 HTML、CSV 与六类 PNG。"""
from __future__ import annotations

from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import statistics

from .metrics import summarize


def load_records(directory):
    path = Path(directory) / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, list):
            result[name] = json.dumps(value, ensure_ascii=False)
        else:
            result[name] = value
    return result


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        if r["kind"] != "generation":
            continue
        key = (r["mode"], r.get("profile", False), r["input_tokens"], r["strategy"], r.get("task"))
        groups[key].append(r)
    output = []
    for (mode, profile, length, strategy, task), rows in sorted(groups.items(), key=lambda x: str(x[0])):
        result = {"mode": mode, "profile": profile, "input_tokens": length, "strategy": strategy,
                  "task": task, "n_requests": len(rows), "n_samples": len({r['sample_id'] for r in rows}),
                  "length_limit_rate": statistics.mean(r["reached_length_limit"] for r in rows),
                  "truncation_rate": None if mode == "performance" else statistics.mean(
                      r.get("truncated", r["termination"] == "max_new_tokens") for r in rows)}
        for field in ("prefill_gpu_ms", "prefill_cpu_ms", "strategy_wall_ms", "decode_gpu_ms", "decode_cpu_ms",
                      "request_wall_ms", "model_ttft_wall_ms", "request_tokens_per_s", "decode_forward_tokens_per_s"):
            result[field] = summarize([r["timing"].get(field) for r in rows])
        for field in ("exact_match", "answer_contains"):
            scores = [(r.get("score") or {}).get(field) for r in rows]
            present = [v for v in scores if v is not None]
            result[field] = {"n": len(present), "mean": statistics.mean(present) if present else None}
        result["kv_after_bytes"] = summarize([r["kv"]["after_bytes"] for r in rows])
        output.append(result)
    return output


def matched_changes(records):
    """仅在同一运行、模式、输入 hash、repeat、seed 内配对，不跨配置强行比较。"""
    def key(r):
        return (r["kind"], r["mode"], r.get("profile", False), r["sample_id"],
                r["input_tokens"], r["repeat"], r["seed"],
                r.get("input", {}).get("sha256"), r.get("reference_sha256"))
    baselines = {key(r): r for r in records if r["strategy"] == "full"}
    output = []
    for r in records:
        if r["strategy"] == "full":
            continue
        b = baselines.get(key(r))
        if not b:
            continue
        row = {"sample_id": r["sample_id"], "input_tokens": r["input_tokens"], "strategy": r["strategy"],
               "repeat": r["repeat"], "kind": r["kind"], "mode": r["mode"],
               "baseline": "full_same_input_repeat_seed", "request_speedup": None,
               "decode_forward_speedup": None, "delta_exact_match": None, "delta_nll": None}
        if r["kind"] == "generation":
            # 只有 CUDA 正式性能模式才输出加速比。诊断、质量、CPU、profiler 不混入。
            if (r["mode"] == "performance" and not r.get("profile") and
                    r["timing"]["gpu_measurement_valid"] and b["timing"]["gpu_measurement_valid"]):
                row["request_speedup"] = b["timing"]["request_wall_ms"] / r["timing"]["request_wall_ms"]
                if r["timing"]["decode_gpu_ms"] and b["timing"]["decode_gpu_ms"]:
                    row["decode_forward_speedup"] = b["timing"]["decode_gpu_ms"] / r["timing"]["decode_gpu_ms"]
            if r.get("score") and b.get("score"):
                if r["score"]["exact_match"] is not None and b["score"]["exact_match"] is not None:
                    row["delta_exact_match"] = r["score"]["exact_match"] - b["score"]["exact_match"]
        else:
            row["delta_nll"] = r["nll"] - b["nll"]
        output.append(row)
    return output


def write_csv(path, rows):
    flattened = [flatten(r) for r in rows]
    fields = sorted({key for row in flattened for key in row})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def build_report(directory, records=None):
    # 从已有原始记录重新生成报告，包含 HTML、CSV、PNG；不依赖模型或 GPU。
    directory = Path(directory)
    records = load_records(directory) if records is None else records
    config = json.loads((directory / "config.resolved.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    summaries, comparisons = aggregate(records), matched_changes(records)
    (directory / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))
    (directory / "comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2, allow_nan=False))
    write_csv(directory / "records.csv", records)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "comparisons.csv", comparisons)
    figures = plot(directory, records, comparisons)
    validation = directory / "full_validation.json"
    validation_text = json.loads(validation.read_text()) if validation.exists() else None
    esc = lambda v: html.escape(str(v))
    def artifact(name, title):
        return f'<a href="{name}">{esc(title)}</a>' if (directory / name).exists() else f'{esc(title)}：未采集'
    artifact_links = " · ".join(artifact(name, title) for name, title in [
        ("config.resolved.json", "有效配置"), ("environment.json", "环境"), ("model.json", "模型与模板"),
        ("code.json", "代码标识"), ("records.jsonl", "JSONL"), ("records.csv", "原始 CSV"),
        ("summary.csv", "聚合 CSV"), ("comparisons.csv", "配对比较")])
    sections = []
    for name, title, caption in figures:
        sections.append(f'<section><h2>{esc(title)}</h2><p>{esc(caption)}</p><img src="{name}" alt="{esc(title)}"></section>')
    table = []
    for s in summaries:
        em = s["exact_match"]
        table.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [s["mode"], s["task"], s["strategy"],
                     s["input_tokens"], s["n_samples"], s["n_requests"],
                     em["mean"] if em["mean"] is not None else "未采集/无标注",
                     f"{s['truncation_rate']:.1%}" if s['truncation_rate'] is not None else "固定步数：不适用", f"{s['request_wall_ms']['median']:.3f}",
                     s["request_wall_ms"]["iqr"]]) + "</tr>")
    outputs = []
    for r in records:
        if r["kind"] == "generation":
            outputs.append(f'<details><summary>{esc(r["sample_id"])} / {esc(r["strategy"])} / repeat {r["repeat"]} / {esc(r["termination"])}</summary><pre>{esc(r["output"])}</pre></details>')
    mode = config["mode"]
    banner = ("CUDA 正式性能实验；仅与本次运行内相同输入/配置/seed 的 full 配对。"
              if mode == "performance" and any(r.get("timing", {}).get("gpu_measurement_valid")
                                               and not r.get("profile") for r in records)
              else "本次结果不能作为正式 GPU 性能结论：质量/诊断/CPU/profiler 数据已隔离。")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVLab 实验报告</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#172b4d;background:#f3f6fa;margin:0 auto;padding:28px;max-width:1200px}}section,header{{background:white;border-radius:12px;padding:22px;margin:18px 0}}h1,h2{{line-height:1.3}}img{{max-width:100%;height:auto}}.notice{{border-left:5px solid #b86c14;padding:12px;background:#fff3d6}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dde4ee;text-align:left;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#176b94}}.scroll{{overflow:auto}}</style>
<header><h1>KVLab · {esc(config['label'])}</h1><p>模式：<strong>{esc(mode)}</strong> · 状态：{esc(status['status'])} · 原始记录：{len(records)}</p>
<p class="notice">{esc(banner)}</p><p>首 token 始终来自完整 prefill 的 logits；一次干预发生在采样首 token 之前，TTFT 包括策略成本。预算仅指 prompt KV，新生成 KV 正常追加。物理索引收集减少 KV 存储与后续 dense attention 的 key 长度，未实现稀疏 kernel；收益必须以实际结果判断。</p>
<p>{esc('；'.join(status.get('warnings', [])))}</p><p>{esc(status.get('error', ''))}</p>
<p>full/no-cache 短前缀预检：{esc(json.dumps(validation_text, ensure_ascii=False)) if validation_text else '未采集'}。{artifact('full_validation.json', '预检原始记录')}；短前缀不代表长上下文正确性或性能已验证。</p>
<p>{artifact_links}</p></header>
<section><h2>测量口径</h2><p>GPU 前向时间来自 CUDA Events，CPU 验证使用 perf_counter，GPU 缺失值保持 null。完整请求从内存中原始 prompt 的 tokenization 前开始，到全部生成 token 搬回 CPU 并解码为文本结束，排除模型加载、数据集读取、评分和文件写入。TTFT 从 prefill 前到策略执行和首 token 采样、搬回 CPU 完成。每步 Events 不逐步同步；自然 EOS 的 host 检查另有标记。</p><p>阶段 GPU interval 含排队/主机发射间隙，策略 wall 包括 CPU 选择和同步。模型前向内包含框架原生 Cache 更新。阶段 Events 与 wall 时间可能重叠，成本图使用并列区间，不做机械求和。显存为 PyTorch allocator 口径，阶段增量是相对阶段开始的基线；reserved 受预热和 allocator 历史影响。KV 字节为实际 tensor 大小，不是 HBM 流量。</p><p>聚合保留中位数、IQR、min/max、标准差；不足 20 次不报告尾延迟，本版统一不估计 p95/p99。准确率以有标注请求为分母，重复请求不等于独立样本；截断率为达到 max_new_tokens 且未遇 EOS 的请求比例。KL=KL(full||strategy)，FP32 log_softmax；首位置 KL 预期为零。</p></section>
<section><h2>真实记录聚合</h2><div class="scroll"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in ['模式','任务','策略','输入 tokens','独立样本','请求 n','EM 均值','截断率','完整请求中位 ms','IQR ms'])}</tr></thead><tbody>{''.join(table) or '<tr><td>未采集</td></tr>'}</tbody></table></div></section>
{''.join(sections)}<section><h2>完整模型输出</h2>{''.join(outputs) or '<p>未采集</p>'}</section>
<section><h2>有效配置</h2><pre>{esc(json.dumps(config, ensure_ascii=False, indent=2))}</pre></section></html>'''
    (directory / "report.html").write_text(document)
    return directory / "report.html"


def plot(directory, records, comparisons):
    # 缓存留在输出目录，避免写入用户 home；无 GUI、无 CDN、可离线打开。
    import os
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".mpl-cache").resolve()))
    # Fontconfig 与 matplotlib 的缓存机制不同，沙箱内都使用可写位置。
    os.environ.setdefault("XDG_CACHE_HOME", str((directory / ".cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130, "font.size": 10})
    gen = [r for r in records if r["kind"] == "generation"]
    perf = [r for r in gen if r["mode"] == "performance" and not r.get("profile")
            and r["timing"]["gpu_measurement_valid"]]
    quality = [r for r in gen if r["mode"] == "quality"]
    figures = []

    def empty(ax, text="Not collected"):
        ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color="#69788e")

    def save(fig, name, title, caption):
        fig.tight_layout()
        fig.savefig(directory / name, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        figures.append((name, title, caption))

    def lines(ax, rows, x, y, xlabel, ylabel):
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            value = y(r)
            if value is not None:
                groups[r["strategy"]][x(r)].append(value)
        for label, pts in sorted(groups.items()):
            xx = sorted(pts)
            ax.plot(xx, [statistics.median(pts[v]) for v in xx], marker="o", label=label)
        if groups:
            ax.legend(fontsize=8)
        else:
            empty(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.18)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lines(axes[0], perf, lambda r: r["input_tokens"], lambda r: r["timing"]["decode_gpu_ms"], "Input tokens", "Decode forward total (ms, median)")
    lines(axes[1], gen, lambda r: r["input_tokens"], lambda r: r["kv"]["after_bytes"]/2**20, "Input tokens", "Physical prompt KV (MiB, median)")
    save(fig, "01-context.png", "上下文长度、decode 与 KV", f"GPU 性能请求 n={len(perf)}；KV 请求 n={len(gen)}。decode 包含 N−1 次前向；KV 为干预后实测 tensor 字节。")

    fig, ax = plt.subplots(figsize=(8, 4))
    # 不同长度/任务独立画线，避免混合不兼容难度的质量点。
    qgroups = defaultdict(list)
    for r in quality:
        qgroups[(r["input_tokens"], r.get("task"))].append(r)
    plotted = False
    for (length, task), rows in qgroups.items():
        methods = defaultdict(lambda: defaultdict(list))
        for r in rows:
            score = (r.get("score") or {}).get("exact_match")
            if score is not None:
                s = r["strategy_config"]
                method = s["name"] + (f" (recent={s.get('recent_window', 0)})" if s["name"] == "random_recent" else "")
                methods[method][r["selection"]["actual_prompt_kv"]].append(score)
        for method, points in methods.items():
            xx = sorted(points)
            ax.plot(xx, [statistics.mean(points[n]) for n in xx], "o-", label=f"{method}, L={length}, {task}")
            plotted = True
    if plotted:
        ax.legend(fontsize=7)
    else:
        empty(ax)
    ax.set(xlabel="Retained prompt KV tokens (full at L)", ylabel="Exact match (mean over labeled requests)", ylim=(-.03, 1.03))
    save(fig, "02-quality.png", "保留预算与任务质量", f"仅自由生成质量模式；请求 n={len(quality)}，独立样本 n={len({r['sample_id'] for r in quality})}。full 横坐标为完整 prompt 长度；缺标注不填零。")

    request_series = {(r["strategy"], r["input_tokens"]) for r in perf}
    fig, ax = plt.subplots(figsize=(10, max(4, 1 + .18 * len(request_series))))
    for length in sorted({r["input_tokens"] for r in perf}):
        # 完整 ID 保留窗口和请求预算；预算封顶后也不混合不同配置。
        subset = [dict(r, strategy=f"{r['strategy']}, L={length}") for r in perf if r["input_tokens"] == length]
        lines(ax, subset, lambda r: r["selection"]["actual_prompt_kv"], lambda r: r["timing"]["request_wall_ms"], "Retained prompt KV tokens", "Complete request (ms, median)")
    if perf:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        empty(ax)
        ax.set(xlabel="Retained prompt KV tokens", ylabel="Complete request (ms)")
    save(fig, "03-request.png", "预算与完整请求耗时", f"仅 CUDA 正式固定步数性能模式，n={len(perf)}；按完整策略 ID（请求预算与最近窗口）及输入长度分组；横轴为实际保留数。边界包含 tokenization、搬运、策略、采样与文本解码。")

    fig, ax = plt.subplots(figsize=(10, 4))
    costgroups = defaultdict(list)
    for r in perf:
        costgroups[(r["strategy"], r["input_tokens"])].append(r)
    if costgroups:
        labels = [f"{s}\nL={n}" for s, n in costgroups]
        x = np.arange(len(labels))
        for i, (field, title) in enumerate([("prefill_gpu_ms", "Prefill GPU"), ("decode_gpu_ms", "Decode forward GPU"),
                                           ("strategy_wall_ms", "Strategy wall"), ("request_wall_ms", "Request wall")]):
            vals = [statistics.median(r["timing"][field] for r in rows) for rows in costgroups.values()]
            ax.bar(x + (i-1.5)*.19, vals, .19, label=title)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.legend(fontsize=8)
    else:
        empty(ax)
    ax.set(ylabel="Elapsed (ms, median)\n(overlapping intervals)")
    save(fig, "04-costs.png", "策略管理与模型执行成本", f"GPU forward 与 wall 区间并列展示，可能重叠，不堆叠相加；n={len(perf)}。原始记录另含采样区间和每步耗时。")

    attention = [(r, d) for r in gen for d in r.get("diagnostics", [])]
    fig, ax = plt.subplots(figsize=(10, max(3.5, min(12, len(attention)*.28))))
    if attention:
        shown = attention[:32]
        matrix = np.array([d["histogram"] for _, d in shown])
        image = ax.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest", extent=(0, 1, len(shown), 0))
        ax.set_yticks(np.arange(len(shown))+.5,
                     [f"{r['strategy']} {r['sample_id']} r{r['repeat']} L{d['layer']} H{d['head']} s{d['step']}" for r, d in shown], fontsize=7)
        fig.colorbar(image, ax=ax, label="Attention probability mass per bin")
        ax.set(xlabel="Original position / (query absolute position + 1)")
    else:
        empty(ax)
    save(fig, "05-attention.png", "选定层/head 的历史访问", f"单 query、原始位置分箱概率质量；重算使用 FP32，两遍分块；已采集 n={len(attention)}，图最多显示 32 条，未采集不补值。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    deltas = defaultdict(list)
    speed = defaultdict(list)
    for c in comparisons:
        if c["delta_nll"] is not None:
            deltas[(c["strategy"], c["input_tokens"])].append(c["delta_nll"])
        if c["request_speedup"] is not None:
            speed[(c["strategy"], c["input_tokens"])].append(c["request_speedup"])
    for ax, points, ylabel in [(axes[0], deltas, "NLL delta vs full\n(nats/token, median)"),
                               (axes[1], speed, "Request speedup\n(full wall / strategy wall)")]:
        if points:
            labels = [f"{s}\nL={length}" for s, length in points]
            ax.bar(labels, [statistics.median(v) for v in points.values()], color="#287d91")
            ax.tick_params(axis="x", labelrotation=20)
        else:
            empty(ax)
        ax.set(ylabel=ylabel)
    save(fig, "06-relative.png", "相对 full 的质量变化与加速比", f"同配置、同输入、同 seed/repeat 配对；NLL 配对 n={sum(map(len, deltas.values()))}，正式 CUDA 请求加速比配对 n={sum(map(len, speed.values()))}。两类运行独立显示，禁止用诊断/质量耗时推断加速。")
    return figures
