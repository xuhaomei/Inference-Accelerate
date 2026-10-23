"""策略扩展点：选择原始 prompt 位置，再由适配器物理收集 KV。"""
from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Selection:
    positions: tuple[int, ...]
    requested_budget: int | None
    actual_budget: int
    fixed_recent: int
    extra_fixed_positions: tuple[int, ...] = ()


def normalize_strategy(strategy):
    """统一有效参数；配置去重、结果 ID 与选择算法共用此规则。"""
    if not isinstance(strategy, dict):
        raise ValueError("策略必须是映射")
    if set(strategy) - {"name", "budget", "recent_window"}:
        raise ValueError("策略包含未知字段")
    name = strategy.get("name")
    if name not in {"full", "recent", "random_recent"}:
        raise ValueError("不支持的策略；扩展需注册并验证")
    budget = strategy.get("budget")
    window = strategy.get("recent_window", 0)
    if type(window) is not int or window < 0:
        raise ValueError("recent_window 必须为非负整数")
    if name == "full":
        if budget is not None:
            raise ValueError("full 不接受预算")
    elif type(budget) is not int or budget < 1:
        raise ValueError("压缩策略 budget 必须为正整数")
    if name != "random_recent" and window != 0:
        raise ValueError(f"{name} 不使用 recent_window，必须为 0 或省略")
    if name == "random_recent" and window > budget:
        raise ValueError("recent_window 必须在预算内")
    return {"name": name, "budget": budget, "recent_window": window}


def strategy_id(strategy):
    """按规范化参数生成身份，保持已有合法配置的 ID 格式。"""
    s = normalize_strategy(strategy)
    return "full" if s["name"] == "full" else f"{s['name']}-b{s['budget']}-w{s['recent_window']}"


def select(length, strategy, seed):
    # full: 选择所有位置；recent: 选择最近的 budget 个位置；random_recent: 选择最近窗口 + 随机位置
    if length < 1:
        raise ValueError("prompt 长度必须为正数")
    strategy = normalize_strategy(strategy)
    name, budget = strategy["name"], strategy.get("budget")
    # full 不需要预算；压缩策略的实际预算封顶到 prompt 长度。
    if name == "full":
        return Selection(tuple(range(length)), None, length, 0)
    actual = min(budget, length)
    # recent 策略选择最近的 actual 个位置；random_recent 策略选择最近窗口 + 随机位置
    if name == "recent":
        positions = tuple(range(length - actual, length))
        return Selection(positions, budget, actual, actual)
    requested_window = strategy["recent_window"]
    window = min(requested_window, actual)
    recent = list(range(length - window, length))
    chosen = random.Random(seed).sample(range(length - window), actual - window)
    return Selection(tuple(sorted(chosen + recent)), budget, actual, window)


# 后续 attention 选择可扩展 select；量化/卸载/稀疏 kernel 应扩展适配器，
# 必须单独定义存储、位置与测量语义，不能把逻辑选择冒充物理加速。
