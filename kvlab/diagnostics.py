"""Diagnostics 是平台的attention 行为采集器，用来观察：在某个生成步骤、某一层、某个 head 中，当前 query 把注意力分配给了哪些历史位置，以及 KV 裁剪后这种分配如何变化。
"""
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class Diagnostics:
    def __init__(self, adapter, config):
        self.adapter, self.config = adapter, config
        self.records, self.skips, self.handles = [], [], []
        self.step, self.absolute_position, self.positions = -1, -1, []
        if any(idx >= len(adapter.model.model.layers) for idx in config["layers"]):
            raise ValueError("诊断层越界")
        if any(h >= adapter.model.config.num_attention_heads for h in config["heads"]):
            raise ValueError("诊断 head 越界")
        for idx in config["layers"]:
            module = adapter.model.model.layers[idx].self_attn
            # 每次指定的 self_attn 模块完成一次 forward() 调用后，就会执行 self.hook，此时Cache已追加当前输入token的KV
            self.handles.append(module.register_forward_hook(self.hook, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def hook(self, module, args, kwargs, output):
        # module：触发 hook 的 self_attn 模块
        # args：传给该模块的位置参数
        # kwargs：传给该模块的关键字参数
        # output：该模块 forward 的返回值
        c = self.config
        if self.step not in c["steps"]:
            return
        hidden = kwargs["hidden_states"]
        if hidden.shape[1] != 1:
            raise RuntimeError("诊断仅允许单 token query；禁止采集全 prefill attention")
        cache = kwargs["past_key_values"]
        keys = cache.layers[module.layer_idx].keys
        nkeys = keys.shape[-2]
        if nkeys > c["max_keys"]:
            self.skips.append({"step": self.step, "layer": module.layer_idx, "reason": "max_keys exceeded"})
            return
        for head in c["heads"]:
            if len(self.records) >= c["max_records"]:
                if not self.skips or self.skips[-1].get("reason") != "max_records exceeded":
                    self.skips.append({"reason": "max_records exceeded"})
                return
            # 当前层输入重算 Q，执行 Qwen3 的 q_norm 和 RoPE。
            # 只构造一个 query 向量；Qwen3 q_norm 在 head_dim 内独立进行。
            q = module.q_norm(module.q_proj(hidden).view(1, 1, -1, module.head_dim)).transpose(1, 2)
            # position_embeddings = rotary_emb(hidden_states, position_ids)
            # cos,sin最后一维是 head_dim；RoPE 只在 head_dim 内独立进行。
            cos, sin = kwargs["position_embeddings"]
            # 对于前后半部分配对的旋转位置编码q_rotated = q * cos + rotate_half(q) * sin
            # 其中rotate_half(q) = torch.cat([-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]], dim=-1)后一半旋转到前半部分并取负号，前半部分旋转到后一半
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            q = q[0, head, 0].float()
            # Grouped-Query Attention：多个 Query 头共享同一个 Key/Value 头
            # 这行代码假设 keys 还保留原始 KV 头数量。 如果此前已经通过 repeat_kv 将 K 扩展到与 Query 相同的头数量，就应直接使用 keys[0, head]，不再除以 num_key_value_groups。
            kh = keys[0, head // module.num_key_value_groups]
            # 分块计算 logsumexp。
            chunks = range(0, nkeys, c["chunk_size"])
            lse = torch.tensor(float("-inf"), device=q.device)
            for start in chunks:
                scores = (kh[start:start+c["chunk_size"]].float() @ q) * module.scaling
                lse = torch.logaddexp(lse, torch.logsumexp(scores, 0))

            histogram = torch.zeros(c["bins"], device=q.device)
            entropy = torch.zeros((), device=q.device)
            top_values = torch.empty(0, device=q.device)
            top_indices = torch.empty(0, dtype=torch.long, device=q.device)
            positions = torch.tensor(self.positions, device=q.device)
            if len(positions) != nkeys:
                raise RuntimeError("诊断原始位置映射与物理 KV 长度不一致")
            for start in chunks:
                end = min(start+c["chunk_size"], nkeys)
                scores = (kh[start:end].float() @ q) * module.scaling
                logp = scores - lse
                p = logp.exp()
                entropy -= (p * logp).sum()
                # 每条 histogram 对应当前 step、当前层、当前 query head 的 attention 分布，但它保存的是按原始序列位置分箱后的概率质量。
                # 横轴：从位置 0 到当前 query 位置，划分为 bins 个区间
                # 每个值：该区间内所有实际保留 KV 的 attention 概率之和。
                bins = (positions[start:end] * c["bins"] // (self.absolute_position+1)).clamp(max=c["bins"]-1)
                histogram.scatter_add_(0, bins, p)
                vals, idx = p.topk(min(c["top_k"], len(p)))
                top_values = torch.cat([top_values, vals])
                top_indices = torch.cat([top_indices, idx + start])
                vals, idx = top_values.topk(min(c["top_k"], len(top_values)))
                top_values, top_indices = vals, top_indices[idx]
            if not torch.isfinite(histogram).all() or not torch.isfinite(entropy):
                raise ValueError("诊断出现 NaN/Inf")
            self.records.append({
                "step": self.step,
                "query_absolute_position": self.absolute_position,
                "layer": module.layer_idx,
                "head": head,
                "kv_head": head // module.num_key_value_groups,
                "physical_keys": nkeys,
                "histogram": histogram.cpu().tolist(),
                "entropy_nats": entropy.item(),
                "top_original_positions": positions[top_indices].cpu().tolist(),
                "top_probabilities": top_values.cpu().tolist(),
                "method": "post_forward_selected_query_recompute_fp32_two_pass_chunked",
                "backend_forward": self.adapter.config["model"]["backend"]})
