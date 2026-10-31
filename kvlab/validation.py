"""固定轨迹的 full 预检。所有观测、同步和 CPU FP64 统计均在正式计时之外。"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import torch
import torch.nn.functional as F


class FullValidationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        reason = report["fatal_error"]["message"] if report["fatal_error"] else "full 与无 Cache 的 logits 超出原有容差"
        super().__init__(
            f"full 预检失败：first_failed_step={report['first_failed_step']}, "
            f"max_abs={report['max_abs_logits_error']}；{reason}；详情见 full_validation.json")


class _FatalValidationError(RuntimeError):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_logits(reference, candidate, token_id, atol, rtol=0.02, top_k=8):
    """旧 FP32 allclose 判定不变；其他统计在 CPU FP64 上计算，不夹断 KL。"""
    a32, b32 = reference.detach().float().flatten(), candidate.detach().float().flatten()
    if a32.shape != b32.shape or not a32.numel():
        raise ValueError("预检 logits 形状不一致或为空")
    a, b = a32.to(device="cpu", dtype=torch.float64), b32.to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {"finite_checks_passed": finite, "legacy_allclose_passed": False,
              "max_abs": None, "mean_abs": None, "rmse": None, "mean_bias": None,
              "centered_rmse": None, "failed_count": None, "failed_fraction": None,
              "max_excess": None, "kl_reference_to_candidate": None, "probability_l1": None,
              "reference_nll": None, "candidate_nll": None,
              "reference_top": None, "candidate_top": None, "top1_equal": None,
              "worst_absolute_errors": [], "worst_violations": [],
              "nonfinite_reference_count": int((~torch.isfinite(a)).sum()),
              "nonfinite_candidate_count": int((~torch.isfinite(b)).sum())}
    if not finite:
        return result  # JSON 中保持 null，不计算 argmax/log_softmax 或把 NaN 夹成 0。
    close = torch.isclose(b32, a32, atol=atol, rtol=rtol).cpu()
    delta = b - a
    error, tolerance = delta.abs(), atol + rtol * a.abs()
    excess = error - tolerance
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    pa, pb = la.exp(), lb.exp()
    result.update(
        legacy_allclose_passed=bool(torch.allclose(b32, a32, atol=atol, rtol=rtol)),
        max_abs=error.max().item(), mean_abs=error.mean().item(), rmse=delta.square().mean().sqrt().item(),
        mean_bias=delta.mean().item(), centered_rmse=(delta-delta.mean()).square().mean().sqrt().item(),
        failed_count=int((~close).sum()), failed_fraction=(~close).double().mean().item(),
        max_excess=excess.max().item(), kl_reference_to_candidate=(pa * (la-lb)).sum().item(),
        probability_l1=(pa-pb).abs().sum().item(),
        reference_nll=-la[token_id].item(), candidate_nll=-lb[token_id].item())

    def top(logits, probabilities):
        values, indices = logits.topk(min(2, logits.numel()))
        index = int(indices[0])
        return {"token_id": index, "logit": values[0].item(), "probability": probabilities[index].item(),
                "top1_top2_logit_margin": (values[0]-values[1]).item() if len(values) > 1 else None}

    result["reference_top"], result["candidate_top"] = top(a, pa), top(b, pb)
    result["top1_equal"] = result["reference_top"]["token_id"] == result["candidate_top"]["token_id"]

    def entry(index):
        return {"token_id": index, "reference_logit": a[index].item(), "candidate_logit": b[index].item(),
                "absolute_error": error[index].item(), "allowed_error": tolerance[index].item(),
                "excess": excess[index].item(), "legacy_violation": not bool(close[index]),
                "reference_probability": pa[index].item(), "candidate_probability": pb[index].item()}

    result["worst_absolute_errors"] = [entry(i) for i in error.topk(min(top_k, len(a))).indices.tolist()]
    violating = torch.where(~close)[0]
    if len(violating):
        indices = violating[excess[violating].topk(min(top_k, len(violating))).indices]
        result["worst_violations"] = [entry(i) for i in indices.tolist()]
    return result


def _tensor_info(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def _identity(adapter):
    m, c = adapter.config["model"], adapter.model.config
    return {"model": m["name"], "model_revision": m["revision"],
            "model_commit": getattr(c, "_commit_hash", None),
            "tokenizer": adapter.tokenizer.name_or_path,
            "tokenizer_revision": m["tokenizer_revision"] or m["revision"],
            "tokenizer_commit": adapter.tokenizer.init_kwargs.get("_commit_hash"),
            "tiny_seed": getattr(adapter, "initialization_seed", None) if m["tiny"] else None,
            "architecture": {key: getattr(c, key, None) for key in (
                "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "max_position_embeddings", "rope_scaling")}}


class _Trace:
    """临时观测真实 rotary/attention/SDPA 边界；退出时恢复所有 hook 和函数。"""
    def __init__(self, adapter, report):
        self.adapter, self.report = adapter, report
        self.current = None
        self.stack = ExitStack()

    def require(self, condition, message, kind="semantic"):
        key = "finite_checks_passed" if kind == "nonfinite" else "semantic_checks_passed"
        if not condition:
            self.report[key] = False
            raise _FatalValidationError(kind, message)
        if self.report[key] is None:
            self.report[key] = True

    def finite(self, tensor, label):
        self.require(bool(torch.isfinite(tensor).all()), f"{label} 含 NaN/Inf", "nonfinite")

    def __enter__(self):
        try:
            handle = self.adapter.model.model.register_forward_pre_hook(self._input, with_kwargs=True)
            self.stack.callback(handle.remove)
            handle = self.adapter.model.model.rotary_emb.register_forward_pre_hook(self._rotary, with_kwargs=True)
            self.stack.callback(handle.remove)
            for index, layer in enumerate(self.adapter.model.model.layers):
                def hook(module, args, kwargs, index=index):
                    self._attention(index, kwargs)
                handle = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
                self.stack.callback(handle.remove)
            original = F.scaled_dot_product_attention

            def sdpa(query, key, value, *args, **kwargs):
                if self.current is not None:
                    self._sdpa(query, key, value, args, kwargs)
                return original(query, key, value, *args, **kwargs)

            self.stack.enter_context(patch.object(F, "scaled_dot_product_attention", sdpa))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def _input(self, module, args, kwargs):
        if self.current is None:
            return
        tensor = kwargs.get("input_ids", args[0] if args else None)
        self.require(isinstance(tensor, torch.Tensor), "未观察到实际 input_ids")
        actual = tensor.detach().cpu().tolist()
        self.current["input_ids"] = actual
        self.require(actual == [self.current["expected_input_token_ids"]], "实际输入偏离固定参考轨迹")

    def _rotary(self, module, args, kwargs):
        if self.current is None:
            return
        positions = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        self.require(positions is not None, "未观察到实际 RoPE position_ids")
        actual = positions.detach().cpu().tolist()
        self.current["position_ids"] = actual
        self.require(actual == [self.current["expected_positions"]], "RoPE position_ids 与固定轨迹不一致")

    def _attention(self, index, kwargs):
        if self.current is None:
            return
        positions = kwargs.get("cache_position")
        self.require(positions is not None, "未观察到实际 cache_position")
        actual = positions.detach().cpu().tolist()
        mask = kwargs.get("attention_mask")
        layer = {"layer": index, "cache_position": actual,
                 "attention_mask": None if mask is None else _tensor_info(mask), "sdpa": None}
        self.current["layers"].append(layer)
        self.require(actual == self.current["expected_positions"], "full 路径的物理 cache_position 不正确")
        if self.adapter.config["model"]["backend"] == "eager":
            self.require(mask is not None and self._mask_matches(mask), "eager 因果 mask 不正确")

    def _mask_matches(self, mask):
        # 本预检只处理 batch=1、无 padding 的完整历史；绝对位置等于物理位置。
        positions = torch.tensor(self.current["expected_positions"], device=mask.device)
        keys = torch.arange(self.current["expected_kv_length"], device=mask.device)
        expected = (keys[None, :] <= positions[:, None])[None, None]
        try:
            expanded = torch.broadcast_to(mask, expected.shape)
            if expanded.dtype == torch.bool:
                return bool(torch.equal(expanded, expected))
            blocked = (expanded == float("-inf")) | (expanded <= torch.finfo(expanded.dtype).min)
            return bool(torch.where(expected, expanded == 0, blocked).all())
        except (RuntimeError, TypeError):
            return False

    def _sdpa(self, query, key, value, args, kwargs):
        self.require(bool(self.current["layers"]), "SDPA 调用未对应到 attention 层")
        mask = kwargs.get("attn_mask", args[0] if args else None)
        causal = kwargs.get("is_causal", args[2] if len(args) > 2 else False)
        layer = self.current["layers"][-1]
        self.require(layer["sdpa"] is None, "同一层出现未预期的多次 SDPA 调用")
        layer["sdpa"] = {"query": _tensor_info(query), "key": _tensor_info(key), "value": _tensor_info(value),
                         "attn_mask": None if mask is None else _tensor_info(mask), "is_causal": bool(causal),
                         "enable_gqa": bool(kwargs.get("enable_gqa", False)),
                         "enabled_backends": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                              "math": torch.backends.cuda.math_sdp_enabled(),
                                              "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                              "cudnn": torch.backends.cuda.cudnn_sdp_enabled()}}
        enabled = layer["sdpa"]["enabled_backends"]
        requested = self.adapter.config["model"]["sdpa_kernel"]
        self.require(enabled[requested] and sum(enabled.values()) == 1, "未严格启用单个指定 SDPA backend")
        c = self.adapter.model.config
        self.require(all(t.ndim == 4 for t in (query, key, value)), "SDPA Q/K/V 维度不正确")
        self.require(query.shape[0] == key.shape[0] == value.shape[0] == 1, "预检只支持 batch=1")
        self.require(query.shape[-2] == len(self.current["expected_positions"])
                     and key.shape[-2] == value.shape[-2] == self.current["expected_kv_length"], "SDPA 序列长度不正确")
        self.require(query.shape[1] == c.num_attention_heads and key.shape[1] == value.shape[1]
                     and key.shape[1] in (c.num_key_value_heads, c.num_attention_heads), "SDPA GQA heads 不正确")
        self.require(query.shape[1] == key.shape[1] or kwargs.get("enable_gqa", False), "未启用不等 head 数需要的 GQA")
        self.require(query.shape[-1] == key.shape[-1] == value.shape[-1]
                     == getattr(c, "head_dim", c.hidden_size // c.num_attention_heads), "SDPA head_dim 不正确")
        self.require(query.dtype == key.dtype == value.dtype == next(self.adapter.model.parameters()).dtype,
                     "SDPA Q/K/V dtype 不一致")
        self.require(query.device == key.device == value.device == next(self.adapter.model.parameters()).device,
                     "SDPA device 不一致")
        self.require(all(t.stride(-1) == 1 for t in (query, key, value)), "SDPA 最后一维不连续")
        for label, tensor in (("query", query), ("key", key), ("value", value)):
            self.finite(tensor, label)
        if mask is None:
            self.require(bool(causal) == (query.shape[-2] > 1), "SDPA is_causal 不符合完整序列/单 token 语义")
        else:
            self.require(not causal and self._mask_matches(mask), "SDPA 显式因果 mask 不正确")

    def cache_state(self, cache, expected_length):
        try:
            self.adapter.check_cache(cache)
        except (TypeError, AttributeError) as exc:
            self.require(False, str(exc))
        self.require(len(cache.layers) == self.adapter.model.config.num_hidden_layers, "Cache 层数不正确")
        self.require(cache.get_seq_length() == expected_length, "Cache 报告的物理长度不正确")
        layers = []
        c = self.adapter.model.config
        dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        for index, layer in enumerate(cache.layers):
            self.require(layer.keys is not None and layer.values is not None, "Cache 含未初始化层")
            state = {"layer": index}
            for name, tensor in (("keys", layer.keys), ("values", layer.values)):
                self.require(list(tensor.shape) == [1, c.num_key_value_heads, expected_length, dim], "KV 形状不正确")
                parameter = next(self.adapter.model.parameters())
                self.require(tensor.dtype == parameter.dtype and tensor.device == parameter.device, "KV dtype/device 不正确")
                self.finite(tensor, f"layer {index} {name}")
                state[name] = {**_tensor_info(tensor), "data_ptr": tensor.data_ptr(), "finite": True}
            layers.append(state)
        return {"physical_length": cache.get_seq_length(), "bytes": self.adapter.cache_bytes(cache), "layers": layers}

    def run(self, state, function):
        self.current = state
        try:
            output = function()
            self.require(list(output.logits.shape) == [1, 1, self.adapter.model.config.vocab_size],
                         "logits_to_keep=1 未返回唯一末尾位置的完整词表 logits")
            self.finite(output.logits, "输出 logits")
            self.require(state["position_ids"] is not None, "未观察到 RoPE 位置")
            self.require(len(state["layers"]) == self.adapter.model.config.num_hidden_layers, "未观察到全部 attention 层")
            if self.adapter.config["model"]["backend"] == "sdpa":
                self.require(all(layer["sdpa"] is not None for layer in state["layers"]), "未观察到全部真实 SDPA 调用")
            state["completed"] = True
            return output
        except Exception as exc:
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            self.current = None


@torch.inference_mode()
def collect_full_validation(adapter, ids):
    from .strategies import select

    v = adapter.config["validation"]
    prefix = list(ids[:v["prefix_tokens"]])
    dtype = next(adapter.model.parameters()).dtype
    atol = 0.08 if dtype == torch.bfloat16 else 0.01 if dtype == torch.float16 else 2e-5
    report = {"schema_version": 2, "passed": False, "semantic_checks_passed": None,
              "finite_checks_passed": None, "legacy_allclose_passed": None,
              "first_failed_step": None, "first_failed_stage": None, "fatal_error": None,
              "planned_steps": v["decode_steps"] + 1, "completed_steps": 0,
              "tested_decode_steps": 0, "prompt_prefix_tokens": len(prefix),
              "source_input_tokens": len(ids),
              "max_abs_logits_error": None, "atol": atol, "rtol": 0.02, "dtype": str(dtype),
              "scope": "short_prefix_correctness_only", "steps": [], "trajectory": None,
              "numeric": "legacy FP32 allclose; CPU FP64 statistics; signed KL; no epsilon clipping",
              "acceptance": "unchanged A_vs_C allclose on decode steps, plus finite/semantic checks; B comparisons diagnostic only",
              "paths": {"A": "platform cached full/no-op", "B": "direct HF cached with inferred positions",
                        "C": "uncached full prefix with explicit 2D all-ones padding mask"}}
    identity = _identity(adapter)
    trajectory = {"schema_version": 1, "complete": False, "prompt_token_ids": prefix,
                  "reference_token_ids": [], "input_sha256": _hash(prefix), "trajectory_sha256": None,
                  "model_identity": identity, "identity_limit": "not a checksum of all weight tensors",
                  "source": {"dtype": str(dtype), "backend": adapter.config["model"]["backend"],
                             "sdpa_kernel": adapter.config["model"]["sdpa_kernel"],
                             "torch_version": torch.__version__, "device": str(adapter.device),
                             "origin": "uncached C greedy; fixed steps ignore EOS"}}
    report["trajectory"] = trajectory
    legacy = []
    active_step, active_stage = None, "trajectory"
    trace = _Trace(adapter, report)

    def ids_valid(tokens):
        return bool(tokens) and all(type(x) is int and 0 <= x < adapter.model.config.vocab_size for x in tokens)

    def remember_failure(step, stage):
        if report["first_failed_stage"] is None:
            report["first_failed_step"], report["first_failed_stage"] = step, stage

    try:
        trace.require(ids_valid(prefix), "预检输入 token IDs 无效")
        trace.require(len(prefix) + v["decode_steps"] <= adapter.model.config.max_position_embeddings,
                      "预检输入和 decode 步数超过模型上下文")
        replay = None
        if v["trajectory_path"]:
            replay = json.loads(Path(v["trajectory_path"]).read_text())
            trace.require(replay.get("schema_version") == 1 and replay.get("complete") is True,
                          "参考轨迹格式不正确或尚未采集完整")
            trace.require(replay.get("model_identity") == identity, "参考轨迹模型/tokenizer 身份不一致")
            trace.require(replay.get("prompt_token_ids") == prefix and replay.get("input_sha256") == _hash(prefix),
                          "参考轨迹输入不一致；必须重放相同 token IDs 和前缀长度")
            reference = replay.get("reference_token_ids", [])
            trace.require(ids_valid(reference) and len(reference) == v["decode_steps"] + 1, "参考轨迹 token 或长度不正确")
            trace.require(replay.get("trajectory_sha256") == _hash([prefix, reference]), "参考轨迹 hash 不匹配")
            trajectory["source"] = {**replay["source"], "replayed_from": str(v["trajectory_path"])}

        cache_a, cache_b = adapter.new_cache(), None
        with trace:
            for step in range(v["decode_steps"] + 1):
                active_step = step
                row = {"step": step, "phase": "prefill" if step == 0 else "decode",
                       "fed_token_id": None if step == 0 else trajectory["reference_token_ids"][step-1],
                       "prediction_absolute_position": len(prefix) + step,
                       "reference_token_id": None, "completed": False, "paths": {}, "comparisons": {}}
                report["steps"].append(row)
                current_ids = prefix + trajectory["reference_token_ids"][:step]

                def state(path, positions):
                    result = {"expected_positions": positions, "expected_kv_length": len(current_ids),
                              "expected_input_token_ids": (current_ids if path == "C" else
                                                           prefix if step == 0 else [row["fed_token_id"]]),
                              "input_ids": None, "position_ids": None, "layers": [], "completed": False}
                    if path != "C":
                        cache = cache_a if path == "A" else cache_b
                        result["cache_before_physical_length"] = cache.get_seq_length() if cache is not None else 0
                        trace.require(result["cache_before_physical_length"] == (0 if step == 0 else len(current_ids)-1),
                                      "前向之前的 Cache 长度不正确")
                    row["paths"][path] = result
                    return result

                def uncached():
                    tensor = adapter.tensor(current_ids)
                    with adapter.backend_context():
                        return adapter.model(input_ids=tensor, attention_mask=torch.ones_like(tensor, dtype=torch.bool),
                                             use_cache=False, logits_to_keep=1,
                                             output_attentions=False, output_hidden_states=False)

                active_stage = "C.forward"
                c = trace.run(state("C", list(range(len(current_ids)))), uncached)
                trace.require(c.past_key_values is None, "无 Cache 参考路径返回了 Cache")
                target = replay["reference_token_ids"][step] if replay else int(c.logits[0, -1].argmax().item())
                trajectory["reference_token_ids"].append(target)
                row["reference_token_id"] = target
                trajectory["trajectory_sha256"] = _hash([prefix, trajectory["reference_token_ids"]])

                positions = list(range(len(prefix))) if step == 0 else [len(current_ids)-1]
                active_stage = "A.forward"
                if step == 0:
                    a = trace.run(state("A", positions), lambda: adapter.prefill(adapter.tensor(prefix), cache_a))
                    before = trace.cache_state(cache_a, len(prefix))
                    original_kv = [(layer.keys.clone(), layer.values.clone()) for layer in cache_a.layers]
                    adapter.intervene(cache_a, select(len(prefix), {"name": "full"}, 0), full=True)
                    after = trace.cache_state(cache_a, len(prefix))
                    trace.require(before == after, "full/no-op 改变了 KV 元数据或存储地址")
                    trace.require(all(torch.equal(keys, layer.keys) and torch.equal(values, layer.values)
                                      for (keys, values), layer in zip(original_kv, cache_a.layers)),
                                  "full/no-op 原地修改了 KV 内容")
                    del original_kv
                    row["paths"]["A"]["full_noop_metadata_unchanged"] = True
                    row["paths"]["A"]["full_noop_values_unchanged"] = True
                else:
                    a = trace.run(state("A", positions), lambda: adapter.decode(
                        adapter.tensor([row["fed_token_id"]]), cache_a, len(current_ids)-1))
                row["paths"]["A"]["cache_after"] = trace.cache_state(cache_a, len(current_ids))

                def hf_cached():
                    tokens = prefix if step == 0 else [row["fed_token_id"]]
                    with adapter.backend_context():
                        return adapter.model(input_ids=adapter.tensor(tokens), past_key_values=cache_b,
                                             use_cache=True, attention_mask=None, logits_to_keep=1,
                                             output_attentions=False, output_hidden_states=False)

                active_stage = "B.forward"
                b = trace.run(state("B", positions), hf_cached)
                cache_b = b.past_key_values
                trace.require(cache_b is not cache_a and cache_b is not None, "A/B 未使用独立 Cache")
                row["paths"]["B"]["cache_after"] = trace.cache_state(cache_b, len(current_ids))
                trace.require(all(x.keys.data_ptr() != y.keys.data_ptr() and x.values.data_ptr() != y.values.data_ptr()
                                  for x, y in zip(cache_a.layers, cache_b.layers)), "A/B 共享 KV 存储")

                active_stage = "metrics"
                for pair, reference_logits, logits in (("A_vs_C", c.logits, a.logits),
                                                       ("A_vs_B", b.logits, a.logits),
                                                       ("B_vs_C", c.logits, b.logits)):
                    metrics = compare_logits(reference_logits, logits, target, atol, top_k=v["top_k"])
                    metrics["candidate_path"], metrics["reference_path"] = pair.split("_vs_")
                    row["comparisons"][pair] = metrics
                    trace.require(metrics["finite_checks_passed"], f"{pair} logits 非有限", "nonfinite")
                if step > 0:
                    metrics = row["comparisons"]["A_vs_C"]
                    legacy.append(metrics["legacy_allclose_passed"])
                    report["max_abs_logits_error"] = max(report["max_abs_logits_error"] or 0, metrics["max_abs"])
                    if not metrics["legacy_allclose_passed"]:
                        remember_failure(step, "A_vs_C.allclose")
                row["completed"] = True
                report["completed_steps"] += 1
                report["tested_decode_steps"] = step
    except Exception as exc:
        remember_failure(active_step, active_stage)
        report["fatal_error"] = {"kind": getattr(exc, "kind", "runtime"), "stage": active_stage,
                                 "step": active_step, "type": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc()}
    trajectory["complete"] = len(trajectory["reference_token_ids"]) == report["planned_steps"]
    report["legacy_allclose_passed"] = (False if False in legacy else
                                         True if len(legacy) == v["decode_steps"] else None)
    if report["completed_steps"] < report["planned_steps"]:
        for key in ("semantic_checks_passed", "finite_checks_passed"):
            if report[key] is True:
                report[key] = None  # 未观察到错误不代表未执行的剩余检查已经通过。
    report["passed"] = (report["fatal_error"] is None and trajectory["complete"]
                        and report["semantic_checks_passed"] is True and report["finite_checks_passed"] is True
                        and report["legacy_allclose_passed"] is True)
    return report
