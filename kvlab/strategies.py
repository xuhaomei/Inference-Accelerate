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


def select(length, strategy, seed):
    # full: 选择所有位置；recent: 选择最近的 budget 个位置；random_recent: 选择最近窗口 + 随机位置
    if length < 1:
        raise ValueError("prompt 长度必须为正数")
    name, budget = strategy["name"], strategy.get("budget")
    # full 策略不需要 budget；recent/random_recent 需要 budget，且不能超过 prompt 长度
    if name == "full":
        return Selection(tuple(range(length)), None, length, 0)
    if name not in {"recent", "random_recent"}:
        raise NotImplementedError(f"未实现策略：{name}")
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget 必须为正数")
    actual = min(budget, length)
    # recent 策略选择最近的 actual 个位置；random_recent 策略选择最近窗口 + 随机位置
    if name == "recent":
        positions = tuple(range(length - actual, length))
        return Selection(positions, budget, actual, actual)
    requested_window = strategy.get("recent_window", 0)
    if not 0 <= requested_window <= budget:
        raise ValueError("最近窗口必须在预算内")
    window = min(requested_window, actual)
    recent = list(range(length - window, length))
    chosen = random.Random(seed).sample(range(length - window), actual - window)
    return Selection(tuple(sorted(chosen + recent)), budget, actual, window)


# 后续 attention 选择可扩展 select；量化/卸载/稀疏 kernel 应扩展适配器，
# 必须单独定义存储、位置与测量语义，不能把逻辑选择冒充物理加速。
