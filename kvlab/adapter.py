"""仅适配 Transformers 4.57.1 Qwen3 + 全注意力 DynamicCache。

重要：position_ids 是未压缩的绝对 RoPE 位置；cache_position 是紧凑存储位置。
只进行单 token decode，所有保留历史与当前 token 都可见，物理 causal mask 正确。
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer


class ByteTokenizer:
    """无需下载的 tiny 测试 tokenizer；不是 Qwen tokenizer，不用于研究结论。"""
    eos_token_id, bos_token_id, pad_token_id = 2, 1, 0
    name_or_path = "builtin-byte-tokenizer-v1"
    init_kwargs = {}
    chat_template = "builtin: <user>prompt</user><assistant>[thinking-prefix]"

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes([i - 3 for i in ids if 3 <= i < 259]).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False, chat_template=None):
        if chat_template:
            raise ValueError("tiny tokenizer 不支持自定义 Jinja 模板")
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        text += "<think>" if enable_thinking else "<think></think>"
        return self.encode(text) if tokenize else text


class Adapter:
    def __init__(self, config):
        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Cache API 已锁定 transformers==4.57.1；升级后须重新验证")
        self.config = config
        m = config["model"]
        # 仅支持 CPU 正确性验证和单 CUDA 卡；不支持多卡、量化、sliding/hybrid attention 或未验证 RoPE 扩展。
        self.device = torch.device(m["device"])
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("第一版只支持 CPU 正确性验证和单 CUDA 卡")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请使用 GPU 机器或显式 CPU tiny 配置")
        if config["mode"] == "performance" and self.device.type != "cuda" and not config["measurement"]["allow_cpu"]:
            raise ValueError("性能模式要求 CUDA；allow_cpu 仅用于正确性 smoke，不产生 GPU 结论")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.device.type == "cpu" and m["backend"] == "sdpa" and m["sdpa_kernel"] != "math":
            raise ValueError("CPU SDPA 必须显式选择 math，禁止静默 fallback")
        
        dtype = getattr(torch, m["dtype"])
        if m["tiny"]:
            self.initialization_seed = torch.initial_seed()
            # 仅用于正确性验证的 tiny Qwen3，随机权重，非研究结论；不下载模型。
            mc = Qwen3Config(vocab_size=259,
                            hidden_size=64,
                            intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16,
                            max_position_embeddings=4096,
                            bos_token_id=1,
                            eos_token_id=2,
                            pad_token_id=0,
                            attention_dropout=0.0)
            mc._attn_implementation = m["backend"]
            self.model = Qwen3ForCausalLM(mc).to(device=self.device, dtype=dtype)
            self.tokenizer = ByteTokenizer()
        else:
            kwargs = {
                "revision": m["revision"],
                "local_files_only": m["local_files_only"],
                "trust_remote_code": False
                }
            mc = AutoConfig.from_pretrained(m["name"], **kwargs)
            self.check_model_config(mc)
            self.model = AutoModelForCausalLM.from_pretrained(
                m["name"], config=mc, dtype=dtype, attn_implementation=m["backend"], **kwargs,
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                m["tokenizer"] or m["name"],
                revision=m["tokenizer_revision"] or m["revision"],
                local_files_only=m["local_files_only"], trust_remote_code=False)
        self.check_model_config(self.model.config)
        self.model.eval().requires_grad_(False)
        self.eos_ids = self.model.generation_config.eos_token_id
        self.eos_ids = set(self.eos_ids if isinstance(self.eos_ids, list) else [self.eos_ids]) - {None}
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.tokenizer.eos_token_id)
        self.check_backend()

    @staticmethod
    def check_model_config(mc):
        # 仅验证 Transformers 4.57.1 Qwen3 + 全注意力 DynamicCache；不支持量化、sliding/hybrid attention 或未验证 RoPE 扩展。
        if mc.model_type != "qwen3":
            raise ValueError(f"未验证模型类型 {mc.model_type}；第一版仅支持稠密 Qwen3")
        if getattr(mc, "quantization_config", None):
            raise ValueError("第一版不支持量化权重模型；须单独实现并验证适配器")
        if getattr(mc, "use_sliding_window", False) or any(
            t != "full_attention" for t in getattr(mc, "layer_types", [])
        ):
            raise ValueError("不支持 sliding/hybrid attention，不能套用紧凑位置语义")
        if getattr(mc, "rope_scaling", None) and mc.rope_scaling.get("rope_type", "default") not in {"default", "linear", "yarn"}:
            raise ValueError("动态/未验证 RoPE 扩展会改变历史旋转语义，第一版拒绝")

    def check_backend(self):
        expected = self.config["model"]["backend"]
        actual = self.model.config._attn_implementation
        if actual != expected or any(layer.self_attn.config._attn_implementation != expected
                                     for layer in self.model.model.layers):
            raise RuntimeError(f"backend 不一致：请求 {expected}，实际 {actual}")

    def backend_context(self):
        if self.config["model"]["backend"] != "sdpa":
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        kernel = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH
            }[self.config["model"]["sdpa_kernel"]]
        # 只启用一个 kernel：不满足约束时报错，不自动转为另一实现。
        return sdpa_kernel(kernel)

    def check_length(self, length):
        """检查输入长度 + 生成长度是否超过模型上下文。"""
        requested = length + self.config["generation"]["max_new_tokens"]
        maximum = self.model.config.max_position_embeddings
        if requested > maximum:
            raise ValueError(f"输入 {length} + 输出 {requested-length} = {requested} 超过模型上下文 {maximum}；未截断或修改 RoPE")

    def tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def new_cache(self):
        return DynamicCache(config=self.model.config)

    @staticmethod
    def check_cache(cache):
        if type(cache) is not DynamicCache or any(type(layer) is not DynamicLayer for layer in cache.layers):
            raise TypeError("仅支持无卸载、全注意力 DynamicCache/DynamicLayer")
        if getattr(cache, "offloading", False):
            raise TypeError("第一版不支持 Cache offloading")

    def prefill(self, ids, cache):
        with self.backend_context():
            return self.model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.arange(ids.shape[1], device=self.device)[None],
                cache_position=torch.arange(ids.shape[1], device=self.device),
                attention_mask=None,
                logits_to_keep=1,
                output_hidden_states=False,
                output_attentions=False
                )

    def decode(self, token, cache, absolute_position):
        self.check_cache(cache)
        # 查询 KV cache 当前的序列长度，也就是已经缓存了多少个 token 位置的 Key 和 Value。 可以通过 layer_idx 指定层，默认查询第 0 层。
        physical_position = cache.get_seq_length()
        with self.backend_context():
            return self.model(
                input_ids=token.reshape(1, 1), 
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.full((1, 1), absolute_position, dtype=torch.long, device=self.device),
                cache_position=torch.full((1,), physical_position, dtype=torch.long, device=self.device),
                # batch=1 无 padding。传全 1 mask 会在 HF 内部 .all()时引入 host 同步；None 让 SDPA 使用正确 causal 语义。
                attention_mask=None,
                logits_to_keep=1,
                output_hidden_states=False,
                output_attentions=False)

    def intervene(self, cache, selection, full=False):
        self.check_cache(cache)
        if full:
            return cache  # full 的 no-op 不复制、不改变任何 cache 状态
        indices = torch.tensor(selection.positions, dtype=torch.long, device=self.device)
        for layer in cache.layers:
            # index_select(-2, ...) 选择 seq_len 维度；-1 是 head_dim，-3 是 batch。
            layer.keys = layer.keys.index_select(-2, indices)
            layer.values = layer.values.index_select(-2, indices)
        return cache

    @staticmethod
    def cache_bytes(cache):
        # 计算 DynamicCache/DynamicLayer 的实际占用字节数；不包括 Python 对象开销。
        Adapter.check_cache(cache)
        return sum(t.numel() * t.element_size() for layer in cache.layers
                   for t in (layer.keys, layer.values) if t is not None)

    def theory_bytes(self, length):
        c = self.model.config
        head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        return 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * length * next(self.model.parameters()).element_size()

    @torch.inference_mode()
    def validate_full(self, ids, *, raise_on_failure=True):
        """先采集固定轨迹的三路径报告；默认保留旧接口的失败抛异常行为。"""
        from .validation import FullValidationError, collect_full_validation
        report = collect_full_validation(self, ids)
        if raise_on_failure and not report["passed"]:
            raise FullValidationError(report)
        return report

    def metadata(self):
        # 使用的模型、tokenizer、配置、参数量、冻结状态、后端实现、SDPA kernel、模板等信息。
        from pathlib import Path
        c = self.model.config
        raw = c.to_dict()
        m = self.config["model"]
        local_artifacts = {}
        for role, location in (("model", m["name"]), ("tokenizer", m["tokenizer"] or m["name"])):
            path = Path(location)
            if not m["tiny"] and path.is_dir():
                manifest = []
                for p in sorted(path.rglob("*")):
                    if not p.is_file() or any(part.startswith(".") for part in p.relative_to(path).parts):
                        continue
                    stat = p.stat()
                    manifest.append({"path": str(p.relative_to(path)), "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if stat.st_size <= 32*2**20 else None})
                local_artifacts[role] = {"absolute_path": str(path.resolve()), "files": manifest,
                                          "large_weight_content_hashed": False}
        return {"type": c.model_type, "class": type(self.model).__name__, "tiny_random_weights": self.config["model"]["tiny"],
                "model_commit": getattr(c, "_commit_hash", None),
                "tokenizer_commit": self.tokenizer.init_kwargs.get("_commit_hash"),
                "model_revision_requested": self.config["model"]["revision"],
                "tokenizer_revision_requested": self.config["model"]["tokenizer_revision"] or self.config["model"]["revision"],
                "model_config": raw, "eos_token_ids": sorted(self.eos_ids),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
                "frozen": not any(p.requires_grad for p in self.model.parameters()),
                "backend_actual": c._attn_implementation, "sdpa_kernel": self.config["model"]["sdpa_kernel"],
                "compile": False, "cuda_graphs": False, "cache": "DynamicCache/DynamicLayer",
                "local_artifacts": local_artifacts,
                "template": self.config["data"]["template"] or self.tokenizer.chat_template,
                "config_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()}
