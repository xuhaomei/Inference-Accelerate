"""实验执行：每个方法/重复独立 prefill，记录原始输出与明确边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import warnings as python_warnings
from datetime import datetime, timezone

import torch

from .adapter import Adapter
from .config import validate
from .data import prepare, read_samples
from .diagnostics import Diagnostics
from .metrics import distribution, summarize, task_score
from .strategies import select, strategy_id
from .timing import Clock, choose_token


def dump(path, obj):
    # 保存 JSON 文件，确保 UTF-8 编码、非 ASCII 字符、缩进和无 NaN
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append(path, obj):
    # 追加 JSONL 文件，确保 UTF-8 编码、非 ASCII 字符和无 NaN
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def reproducible(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 不强制会改变性能路径的 deterministic kernel；环境元数据记录该选择。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def generate(adapter, sample, config, target, strategy, seed, collect=False):
    clock = Clock(adapter.device)
    clock.sync()
    request_start = time.perf_counter()
    start = time.perf_counter()
    ids, input_meta = prepare(adapter.tokenizer, sample, config, target)
    adapter.check_length(len(ids))
    # 记录 tokenization 耗时，避免与 prefill/forward 混合
    tokenization_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    # input_transfer 在设备上执行，成本留在完整请求中，和模型 forward Events 分开。
    with clock.span("input_transfer"):
        input_ids = adapter.tensor(ids)
    clock.sync()
    # input_transfer_ms 仅记录 CPU wall clock
    input_transfer_ms = (time.perf_counter()-start)*1000
    cache = adapter.new_cache()
    generator = torch.Generator(device=adapter.device).manual_seed(seed)
    diag = Diagnostics(adapter, config["diagnostics"]) if collect else None
    try:
        prefill_base = clock.memory_start()
        clock.sync()
        model_start = time.perf_counter()
        with clock.span("prefill"):
            out = adapter.prefill(input_ids, cache)
        prefill_memory = clock.memory_end(prefill_base)
        kv_before = adapter.cache_bytes(cache)
        strategy_start = time.perf_counter()
        with clock.span("strategy"):
            selection = select(len(ids), strategy, seed)
            adapter.intervene(cache, selection, full=strategy["name"] == "full")
        clock.sync()
        strategy_wall_ms = (time.perf_counter()-strategy_start)*1000
        kv_after = adapter.cache_bytes(cache)
        with clock.span("sampling"):
            # 采样下一个token
            first = choose_token(out.logits, config, generator)
        # 首 token materialize 是 TTFT 的终点，位于干预与采样之后。
        first_value = int(first.item())
        clock.sync()
        ttft_ms = (time.perf_counter()-model_start)*1000
        tokens = [first]
        stop = "eos" if not config["generation"]["ignore_eos"] and first_value in adapter.eos_ids else None
        del out
        decode_base = clock.memory_start()
        clock.sync()
        decode_start = time.perf_counter()
        with clock.span("decode_phase"):
            for step in range(1, config["generation"]["max_new_tokens"]):
                if stop:
                    break
                if diag:
                    diag.step = step
                    diag.absolute_position = len(ids) + step - 1
                    diag.positions = list(selection.positions) + list(range(len(ids), len(ids)+step))
                with clock.span("decode_forward"):
                    out = adapter.decode(tokens[-1], cache, len(ids)+step-1)
                with clock.span("sampling"):
                    token = choose_token(out.logits, config, generator)
                tokens.append(token)
                if not config["generation"]["ignore_eos"] and int(token.item()) in adapter.eos_ids:
                    stop = "eos"
                del out
        decode_memory = clock.memory_end(decode_base)
        clock.sync()
        decode_wall_ms = (time.perf_counter()-decode_start)*1000
        model_wall_ms = (time.perf_counter()-model_start)*1000
        output_ids = torch.cat(tokens).cpu().tolist()
        text = adapter.tokenizer.decode(output_ids, skip_special_tokens=True)
        request_ms = (time.perf_counter()-request_start)*1000
        forward_times = clock.values("decode_forward")
        decode_ms = sum(forward_times)
        steps = len(output_ids)-1
        # TODO:每个指标要检查是否统计正确
        timing = {
            "timing_device": "cuda_events" if clock.cuda else "cpu_perf_counter",
            "gpu_measurement_valid": clock.cuda and config["mode"] == "performance" and not config["model"]["tiny"],
            "tokenization_ms": tokenization_ms,
            "input_transfer_wall_ms": input_transfer_ms,
            "input_transfer_gpu_ms": sum(clock.values("input_transfer")) if clock.cuda else None,
            "prefill_gpu_ms": sum(clock.values("prefill")) if clock.cuda else None,
            "prefill_cpu_ms": sum(clock.values("prefill")) if not clock.cuda else None,
            "strategy_gpu_interval_ms": sum(clock.values("strategy")) if clock.cuda else None,
            "strategy_wall_ms": strategy_wall_ms,
            "decode_gpu_ms": decode_ms if clock.cuda else None,
            "decode_cpu_ms": decode_ms if not clock.cuda else None,
            "decode_step_gpu_ms": forward_times if clock.cuda else [],
            "decode_step_cpu_ms": forward_times if not clock.cuda else [],
            "decode_step_forward_ms_summary": summarize(forward_times),
            "decode_phase_gpu_interval_ms": sum(clock.values("decode_phase")) if clock.cuda else None,
            "sampling_gpu_interval_ms": sum(clock.values("sampling")) if clock.cuda else None,
            "decode_wall_ms": decode_wall_ms, "model_ttft_wall_ms": ttft_ms,
            "model_request_wall_ms": model_wall_ms, "request_wall_ms": request_ms,
            "decode_forward_steps": steps,
            "decode_forward_tokens_per_s": steps*1000/decode_ms if decode_ms > 0 and steps else None,
            "decode_wall_tokens_per_s": steps*1000/decode_wall_ms if decode_wall_ms > 0 and steps else None,
            "request_tokens_per_s": len(output_ids)*1000/request_ms,
            "eos_host_check_per_step": not config["generation"]["ignore_eos"]}
        return {
            "kind": "generation",
            "strategy": strategy_id(strategy),
            "strategy_config": strategy,
            "seed": seed,
            "input_tokens": len(ids),
            "input_ids": ids,
            "input": input_meta,
            "output_ids": output_ids,
            "output": text,
            "generated_tokens": len(output_ids),
            "termination": stop or "max_new_tokens",
            "reached_length_limit": len(output_ids) == config["generation"]["max_new_tokens"],
            "truncated": None if config["generation"]["ignore_eos"] else stop is None,
            "ignore_eos": config["generation"]["ignore_eos"],
            "score": task_score(sample, text) if config["mode"] != "performance" else None,
            "selection": {"positions": list(selection.positions), "requested_budget": selection.requested_budget,
            "actual_prompt_kv": selection.actual_budget,
            "fixed_recent": selection.fixed_recent,
            "extra_fixed_positions": list(selection.extra_fixed_positions),
            "intervention": "post_prefill_once", "representation": "physical_gather_dynamic_cache",
            "first_token_source": "unmodified_full_prefill_logits",
            "next_decode_absolute_position": len(ids), "next_decode_physical_position": selection.actual_budget},
            "kv": {
                "before_bytes": kv_before,
                "after_bytes": kv_after,
                "final_bytes": adapter.cache_bytes(cache),
                "theory_after_bytes": adapter.theory_bytes(selection.actual_budget),
                "actual_minus_theory_bytes": kv_after-adapter.theory_bytes(selection.actual_budget),
                "final_physical_tokens": cache.get_seq_length(),
                "final_generated_kv_tokens": max(0, len(output_ids)-1),
                "meaning": "tensor_numel_times_element_size_not_HBM_traffic"},
                "memory": {"prefill": prefill_memory, "decode": decode_memory},
                "timing": timing,
                "diagnostics": diag.records if diag else [],
                "diagnostic_skips": diag.skips if diag else []}
    finally:
        if diag:
            diag.close()


@torch.inference_mode()
def teacher_forced(adapter, ids, reference_tokens, strategy, config, seed):
    """两个独立 KV 状态，逐步喂入完全相同的 full continuation；不留全 vocab 历史。"""
    reference_cache, cache = adapter.new_cache(), adapter.new_cache()
    tensor = adapter.tensor(ids)
    ref_logits = adapter.prefill(tensor, reference_cache).logits
    logits = adapter.prefill(tensor, cache).logits
    selection = select(len(ids), strategy, seed)
    adapter.intervene(cache, selection, full=strategy["name"] == "full")
    local, nll, base_nll, all_kl = [], [], [], []
    for pos, token_id in enumerate(reference_tokens):
        metrics = distribution(ref_logits, logits, token_id, config["quality"]["save_top_k"])
        nll.append(metrics["nll"])
        base_nll.append(metrics["baseline_nll"])
        all_kl.append(metrics["kl_full_to_strategy"])
        if pos in config["quality"]["positions"]:
            local.append({"continuation_position": pos, "reference_token_id": token_id, **metrics})
        if pos + 1 < len(reference_tokens):
            token = adapter.tensor([token_id])
            ref_logits = adapter.decode(token, reference_cache, len(ids)+pos).logits
            logits = adapter.decode(token, cache, len(ids)+pos).logits
    if not reference_tokens:
        raise ValueError("参考 continuation 不能为空")
    avg = lambda xs: sum(xs)/len(xs)
    return {
        "kind": "teacher_forced",
        "strategy": strategy_id(strategy),
        "strategy_config": strategy,
        "seed": seed,
        "input_tokens": len(ids), 
        "reference_token_ids": reference_tokens,
        "reference_sha256": hashlib.sha256(json.dumps(reference_tokens).encode()).hexdigest(),
        "reference_length": len(reference_tokens), "nll": avg(nll), "baseline_nll": avg(base_nll),
        "delta_nll": avg(nll)-avg(base_nll), 
        "mean_kl_full_to_strategy": avg(all_kl),
        "local": local, 
        "first_token_source": "unmodified_full_prefill_logits",
        "numeric": "FP32 log_softmax, KL(full||strategy), no epsilon probability clipping",
        "scope": "fixed_reference_trajectory_not_free_generation_quality"}


def run(config, source_path, overrides=(), profile=False):
    # Python 调用入口也校验：重复 ID 在加载模型、创建目录前失败。
    validate(config)
    from .environment import environment, code_identity
    from .report import build_report
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + config["label"]
    if Path(config["label"]).name != config["label"] or config["label"] in {".", "..", ""}:
        raise ValueError("label 必须是单一目录名")

    # 创建实验目录，保存配置、环境、源代码快照和初始状态
    directory = Path(config["output_dir"]) / run_name
    directory.mkdir(parents=True, exist_ok=False)
    dump(directory / "config.resolved.json", config)
    dump(directory / "config.source.json", {
        "path": str(Path(source_path).resolve()),
        "contents": Path(source_path).read_text(),
        "overrides": list(overrides)})
    dump(directory / "environment.json", environment())
    dump(directory / "code.json", code_identity())
    records, warnings = [], []
    if config["mode"] != "performance":
        warnings.append("质量和诊断运行的耗时不能用于正式性能结论")
    if config["model"]["tiny"]:
        warnings.append("随机 tiny Qwen3 仅验证正确性；输出与性能不能推断 Qwen3-8B")
    if config["measurement"]["repeats"] < 20:
        warnings.append("重复数不足，不输出 p95/p99 尾延迟")
    status = {
        "status": "running",
        "profile": profile,
        "warnings": warnings
        }
    dump(directory / "status.json", status)

    profiler = None
    warning_context = python_warnings.catch_warnings(record=True)
    captured_warnings = warning_context.__enter__()
    try:
        reproducible(config["seed"])
        adapter = Adapter(config)
        dump(directory / "model.json", adapter.metadata())
        samples = read_samples(config["data"]["path"], config["data"]["ids"], config["data"]["limit"])
        dump(directory / "samples.json", samples)
        # 全量预检在任何正式测量前完成，避免部分扫描后才发现超长。
        cases = []
        for sample in samples:
            for target in config["data"]["input_lengths"]:
                ids, _ = prepare(adapter.tokenizer, sample, config, target)
                adapter.check_length(len(ids))
                cases.append((sample, target, ids))
        # 检测第一个样本使用KV Cache和不使用KV Cache的输出是否一致
        dump(directory / "full_validation.json", adapter.validate_full(list(cases[0][2])))
        if profile:
            warnings.append("独立 profiler 运行，全部耗时不可用于正式性能结论")
        strategies = sorted(config["strategies"], key=lambda s: s["name"] != "full")
        for case_index, (sample, target, ids) in enumerate(cases):
            for strategy in strategies:
                for _ in range(config["measurement"]["warmup"]):
                    generate(adapter, sample, config, target, strategy, config["seed"], collect=False)
            # 各方法全部预热后再开始测量；不进行 per-step empty_cache。
            if profile and profiler is None:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if adapter.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
                profiler.__enter__()
            for repeat in range(config["measurement"]["repeats"]):
                seed = config["seed"] + case_index*100003 + repeat
                results = []
                # 轮换方法顺序减少固定先后偏差；full 参考由结果映射获取。
                order = strategies[repeat % len(strategies):] + strategies[:repeat % len(strategies)]
                for strategy in order:
                    result = generate(adapter, sample, config, target, strategy, seed,
                                      collect=config["mode"] == "diagnostic")
                    result.update(
                        sample_id=sample["id"],
                        task=sample.get("task"),
                        repeat=repeat,
                        mode=config["mode"],
                        profile=profile
                        )
                    records.append(result)
                    results.append(result)
                    append(directory / "records.jsonl", result)
                if config["mode"] == "quality" and config["quality"]["teacher_forcing"]:
                    reference = next(r for r in results if r["strategy"] == "full")["output_ids"]
                    for strategy in strategies:
                        r = teacher_forced(adapter, ids, reference, strategy, config, seed)
                        r.update(sample_id=sample["id"], task=sample.get("task"), repeat=repeat,
                                 mode=config["mode"], profile=profile)
                        records.append(r)
                        append(directory / "records.jsonl", r)
                # 每个重复完成后才输出一次简洁进度，性能区间内无逐步日志。
                print(f"完成 sample={sample['id']} tokens={len(ids)} repeat={repeat+1}", flush=True)
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(str(directory / "trace.json"))
        status["runtime_warnings"] = [{
            "category": w.category.__name__, "message": str(w.message),
            "filename": w.filename, "lineno": w.lineno} for w in captured_warnings]
        warnings.extend(f"{w.category.__name__}: {w.message}" for w in captured_warnings)
        warning_context.__exit__(None, None, None)
        dump(directory / "status.json", status)
        build_report(directory, records)
        print(f"实验目录：{directory.resolve()}", flush=True)
    return directory
