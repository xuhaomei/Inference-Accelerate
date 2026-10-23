"""配置合并严格检查未知字段；不隐式截断输入或修改 RoPE。"""
from __future__ import annotations

import copy
from pathlib import Path

from .strategies import normalize_strategy, strategy_id

# 默认配置，允许 extends 覆盖，允许命令行覆盖
DEFAULTS = {
    "mode": "quality",
    "seed": 42,
    "label": "experiment",
    "output_dir": "outputs",
    "model": {
        "name": "Qwen/Qwen3-8B",
        "tokenizer": None,
        "revision": "main",
        "tokenizer_revision": None,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "backend": "sdpa",
        "sdpa_kernel": "flash",
        "local_files_only": False,
        "tiny": False
        },
    "data": {
        "path": "data/synthetic.jsonl",
        "ids": [],
        "limit": None,
        "input_lengths": [None],
        "padding_text": " Irrelevant background record.",
        "chat_template": True,
        "template": None,
        "thinking": False
        },
    "generation": {
        "max_new_tokens": 256,
        "mode": "greedy",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "ignore_eos": False
        },
    "strategies": [{
        "name": "full",
        "budget": None,
        "recent_window": 0
        }],
    "intervention": "post_prefill_once",
    "measurement": {
        "warmup": 1,
        "repeats": 3,
        "allow_cpu": False
        },
    "quality": {
        "teacher_forcing": True,
        "positions": [0, 1, 3, 7, 15, 31],
        "save_top_k": 8
        },
    "diagnostics": {
        "layers": [0],
        "heads": [0],
        "steps": [1, 4, 16],
        "bins": 64,
        "top_k": 8,
        "chunk_size": 2048,
        "max_keys": 65536,
        "max_records": 32
        },
}


def merge(base, patch, prefix=""):
    """递归合并两个配置字典，严格检查未知字段"""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"未知配置字段：{prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} 必须是映射")
            result[key] = merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def load(path, overrides=()):
    """加载 YAML 配置文件，支持 extends 和覆盖，严格检查字段"""
    import yaml
    path = Path(path).resolve()
    visited = set()

    def read(p):
        if p in visited:
            raise ValueError("配置 extends 存在循环")
        visited.add(p)
        doc = yaml.safe_load(p.read_text()) or {}
        parent = doc.pop("extends", None)
        base = read((p.parent / parent).resolve()) if parent else copy.deepcopy(DEFAULTS)
        visited.remove(p)
        return merge(base, doc)

    config = read(path)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError("覆盖格式应为 dotted.key=value")
        node = patch = {}
        parts = key.split(".")
        for part in parts[:-1]:
            node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
        config = merge(config, patch)
    validate(config)
    return config


def validate(c):
    if c["mode"] not in {"performance", "quality", "diagnostic"}:
        raise ValueError("mode 必须是 performance/quality/diagnostic")
    m, g, d = c["model"], c["generation"], c["data"]
    if m["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("不支持的精度")
    if m["backend"] not in {"eager", "sdpa"}:
        raise ValueError("第一版仅支持 eager/sdpa；不允许隐式 backend fallback")
    if m["sdpa_kernel"] not in {"flash", "efficient", "math"}:
        raise ValueError("sdpa_kernel 必须显式指定 flash/efficient/math")
    if c["mode"] == "performance" and m["backend"] != "sdpa":
        raise ValueError("正式性能模式要求 sdpa；诊断 eager 数据不能用于性能结论")
    if c["mode"] == "performance" and str(m["device"]).startswith("cuda") and m["sdpa_kernel"] == "math":
        raise ValueError("正式 CUDA 性能模式要求 flash/efficient；math 仅用于质量/诊断或 CPU 流程检查")
    if c["mode"] == "performance" and not g["ignore_eos"]:
        raise ValueError("性能扫描必须显式 ignore_eos=true，以固定生成步数")
    if c["mode"] != "performance" and g["ignore_eos"]:
        raise ValueError("质量/诊断模式必须遵循正常 EOS")
    if c["intervention"] != "post_prefill_once":
        raise ValueError("第一版仅支持 post_prefill_once")
    if g["mode"] not in {"greedy", "sample"} or g["temperature"] <= 0:
        raise ValueError("无效生成模式或 temperature")
    if not 0 < g["top_p"] <= 1 or g["top_k"] < 0 or g["max_new_tokens"] < 1:
        raise ValueError("无效采样参数或生成长度")
    if c["measurement"]["warmup"] < 0 or c["measurement"]["repeats"] < 1:
        raise ValueError("warmup>=0，repeats>=1")
    if not d["input_lengths"] or any(n is not None and (not isinstance(n, int) or n < 1)
                                     for n in d["input_lengths"]):
        raise ValueError("input_lengths 必须是正整数/null 列表")
    if d["limit"] is not None and d["limit"] < 1:
        raise ValueError("data.limit 必须为正数")
    if not isinstance(c["strategies"], list) or not c["strategies"]:
        raise ValueError("strategies 必须为非空列表，并包含 full 基线")
    seen = set()
    normalized = []
    for s in c["strategies"]:
        s = normalize_strategy(s)
        identity = strategy_id(s)
        if identity in seen:
            raise ValueError(f"策略配置重复（结果 ID: {identity}）")
        seen.add(identity)
        normalized.append(s)
    if not any(s["name"] == "full" for s in normalized):
        raise ValueError("每次实验必须包含 full 基线")
    diag = c["diagnostics"]
    if any(diag[k] < 1 for k in ("bins", "chunk_size", "max_keys", "max_records", "top_k")):
        raise ValueError("诊断限制必须为正数")
    for key in ("layers", "heads", "steps"):
        if any(not isinstance(x, int) or x < 0 for x in diag[key]):
            raise ValueError("诊断索引必须非负")
        if len(set(diag[key])) != len(diag[key]):
            raise ValueError("诊断索引不能重复")
    if 0 in diag["steps"]:
        raise ValueError("诊断 steps 从 1 开始；首 token 来自 prefill，不采集完整 prefill attention")
    if c["quality"]["save_top_k"] < 1 or any(p < 0 for p in c["quality"]["positions"]):
        raise ValueError("无效质量采集参数")
    # 有效配置持久化同一份规范化参数，避免配置与结果身份不一致。
    c["strategies"] = normalized
