"""阶段同步，decode 每步只记录 Events，不为计时逐步 synchronize。"""
from contextlib import contextmanager
import time

import torch


class Clock:
    def __init__(self, device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.spans = {}

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def span(self, name):
        if self.cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.spans.setdefault(name, []).append((start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self.spans.setdefault(name, []).append((time.perf_counter() - start) * 1000)

    def values(self, name):
        entries = self.spans.get(name, [])
        if self.cuda:
            self.sync()
        return [a.elapsed_time(b) for a, b in entries] if self.cuda else entries

    def memory_start(self):
        if not self.cuda:
            return None
        self.sync()
        base = {"allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device)}
        torch.cuda.reset_peak_memory_stats(self.device)
        return base

    def memory_end(self, base):
        if not self.cuda:
            return None
        self.sync()
        return {"baseline_allocated_bytes": base["allocated"], "baseline_reserved_bytes": base["reserved"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)-base["allocated"],
                "incremental_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)-base["reserved"]}


def choose_token(logits, config, generator):
    """采样在设备执行；成本留在完整请求中，和模型 forward Events 分开。"""
    g = config["generation"]
    scores = logits[:, -1].float()
    if g["mode"] == "greedy":
        return scores.argmax(-1)
    scores = scores / g["temperature"]
    # TODO: 采样过程后面要清楚分析
    if g["top_k"]:
        threshold = scores.topk(min(g["top_k"], scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if g["top_p"] < 1:
        ordered, indices = scores.sort(descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > g["top_p"]
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = scores.scatter(1, indices, ordered.masked_fill(remove, float("-inf")))
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).reshape(1)
