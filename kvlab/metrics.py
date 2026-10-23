"""所有分布指标使用 FP32 log_softmax；KL 方向为 full || intervention。"""
import math
import re
import statistics


def distribution(reference_logits, logits, token_id, top_k=8):
    import torch
    a, b = reference_logits.float().flatten(), logits.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("logits 含 NaN/Inf")
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    pa, pb = la.exp(), lb.exp()
    kl = (pa * (la - lb)).sum().item()
    result = {
        "kl_full_to_strategy": max(0.0, kl),
        "nll": -lb[token_id].item(),
        "baseline_nll": -la[token_id].item(),
        "logits_rmse": (a-b).square().mean().sqrt().item(),
        "distribution_l1": (pa-pb).abs().sum().item()
    }
    values, indices = pb.topk(min(top_k, pb.numel()))
    result["strategy_top_tokens"] = [{"id": i, "probability": p} for i, p in zip(indices.tolist(), values.tolist())]
    values, indices = pa.topk(min(top_k, pa.numel()))
    result["baseline_top_tokens"] = [{"id": i, "probability": p} for i, p in zip(indices.tolist(), values.tolist())]
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
        raise ValueError("质量指标含 NaN/Inf")
    return result


def task_score(sample, text):
    ref = sample.get("reference")
    if ref is None:
        return {"exact_match": None, "answer_contains": None, "reason": "无标注，不计算准确率"}
    refs = ref if isinstance(ref, list) else [ref]
    if not refs or not all(isinstance(r, str) for r in refs):
        raise ValueError("reference 必须为字符串或非空字符串列表")
    # 去掉完整 thinking 区间；不把思考中偶然出现答案当作最终答案。
    # 模板可能已在 prompt 末尾打开 <think>，生成文本中只有闭合标签。
    answer = text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text.strip()
    if "<think>" in answer:
        answer = ""  # 未闭合思考没有最终答案
    normalize = lambda x: " ".join(x.strip().casefold().split())
    return {"exact_match": float(any(normalize(answer) == normalize(r) for r in refs)),
            "answer_contains": float(any(re.search(r"(?<!\w)" + re.escape(r) + r"(?!\w)", answer)
                                         is not None for r in refs)),
            "reason": None}


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "stdev": None, "iqr": None}
    if not all(math.isfinite(v) for v in values):
        raise ValueError("聚合输入包含 NaN/Inf")
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "iqr": q[2] - q[0]}
