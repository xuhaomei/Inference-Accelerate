# 测量边界与解释

所有稳态记录都在模型加载之后。每个方法先完成配置指定次数的完整请求预热，再重新初始化 Cache 测量；方法顺序按 repeat 轮换。预热不进入 records。默认不启用 compile/CUDA Graph，不存在待摊销的编译时间。不同策略不共享预填充 Cache，以避免干预污染和不公平省略成本。

| 字段 | 单位与边界 |
|---|---|
| `tokenization_ms` | CPU wall ms，内存 prompt 的模板、编码、精确长度前缀构造与长度检查；不含文件读取 |
| `input_transfer_wall_ms` | CPU 张量构造、搬到目标设备与完成同步的 wall ms |
| `input_transfer_gpu_ms` | 对应 CUDA Event 区间，CPU 构造/发射导致的间隙可包含其中；不是纯 PCIe 带宽测量 |
| `prefill_gpu_ms` | CUDA Events 包围模型 prefill，包含原生 Cache 创建/更新；只计算最后位置 logits |
| `strategy_wall_ms` | CPU 索引选择、设备索引构造、所有层 K/V index_select 与完成同步 |
| `strategy_gpu_interval_ms` | 该策略操作 Event 区间，可能包含 CPU 发射间隙，不能当成纯 kernel 总时长 |
| `decode_gpu_ms` | 所有单 token 模型前向 Events 之和，包含原生 Cache append、mask、模型计算；不含显式采样 |
| `decode_step_gpu_ms` | 上述各步 Event 时间数组；不逐步强制同步 |
| `decode_phase_gpu_interval_ms` | 首 token 已就绪后的 decode 循环整体 GPU Event 区间，含采样与发射间隙 |
| `sampling_gpu_interval_ms` | 首 token 和各步采样 Event 区间之和，不从完整请求中删去 |
| `decode_wall_ms` | 首 token 就绪后，从 decode 前阶段同步结束到整个 decode 完成同步的 wall |
| `model_ttft_wall_ms` | prefill 前至完整策略执行、首 token 采样并搬回 CPU/同步结束；不含 tokenizer/输入搬运 |
| `model_request_wall_ms` | prefill 前至所有 decode 完成同步；包括阶段测量和策略成本，不含输出文本解码 |
| `request_wall_ms` | tokenization 前至全部输出 token 搬回 CPU 并解码为字符串；不含模型加载、文件读取、任务评分、报告和磁盘输出 |
| `decode_forward_tokens_per_s` | 实际 decode 前向步数 ×1000 / decode forward 总 ms；不是端到端 token 吞吐 |
| `decode_wall_tokens_per_s` | 实际 decode 前向步数 ×1000 / decode_wall_ms |
| `request_tokens_per_s` | 全部实际输出 token 数 ×1000 / 完整请求 wall ms |

CPU 验证只填 `prefill_cpu_ms/decode_cpu_ms/decode_step_cpu_ms`；GPU 字段 null，`gpu_measurement_valid=false`。质量/诊断运行即使发生 CUDA 计时，也不可成为正式性能样本。独立 profiler 的记录 `profile=true`，报告排除其加速比。

首 token 来自未干预的完整 prefill logits。干预在它被采样/交付之前执行，因此 TTFT 包含实际策略开销；不能声称首个 token 已验证了裁剪影响。N 个输出 token 只需 N−1 个 decode forward；先遇 EOS 时步数更少。固定步数实验显式忽略 EOS，性能循环内不 `.item()` 检查 EOS；第一个 token 的 materialization 是 TTFT 必要同步。质量/诊断模式为了正常 EOS 会在每步读取 token，`eos_host_check_per_step=true`，这种成本仍包含在 wall-clock 中。

CUDA Event 在当前设备/默认流记录。每个 decode forward 仅插入首尾 Events；统一在 decode 阶段末同步，然后读取所有 Events。不为了记录每步时间强制同步。不调用 per-step empty_cache。prefill、干预和 decode 边界有必要同步，用于清楚分离阶段内存峰值和 TTFT；这些同步的真实成本仍包含在完整请求中。Event 区间可能含 GPU idle、CPU 发射间隙，wall 区间与 GPU 区间也可能重叠。不要相加得到总请求，更不要把独立 GPU kernel 时长与 API wall 混为一谈。

显存 `prefill/decode` 各自先同步、记录 baseline allocated/reserved、reset_peak_memory_stats，阶段末同步读取峰值；增量峰值 = peak − baseline。decode baseline 在干预和首 token 就绪之后获取，包含模型参数和保留 KV。reserved 取决于预热和 allocator 历史，不强制清空 allocator，不能据 reserved 下降与否判断物理 KV 是否变小。策略收集可能临时同时持有一层的新旧张量；其峰值未单列为正式显存阶段，实际成本保留在请求和策略时间中。

KV 实测 = 所有 Cache 层 K/V tensor 的 `numel * element_size`。理论 = `2 * layers * batch(1) * num_KV_heads * head_dim * physical_tokens * dtype_bytes`，按 GQA KV heads 计数，保存实测差值。该数不含 allocator 对齐、临时算子内存、索引或模型参数，更不是 HBM 流量。DynamicCache append 内部 torch.cat 的数据复制成本包含在 forward 中；这会影响策略的实际收益，应与未来静态/分页 Cache 单独比较。

每组保留全部重复，JSON/CSV 输出 n、中位数、IQR、最小/最大与标准差。单样本标准差为 null。本版不输出 p95/p99，也不从同一请求中非独立的 token 步数伪造请求级尾延迟。图中 decode 时间是全部 decode forward 总时间；每步原始分布留在 records。

## 独立分析工具

```bash
python -m kvlab profile --config configs/gpu-smoke.yaml
nsys profile --trace=cuda,nvtx,osrt --output=kv-smoke \
  python -m kvlab run --config configs/gpu-smoke.yaml
ncu --set basic --target-processes all --launch-count 20 \
  python -m kvlab run --config configs/gpu-smoke.yaml
```

Nsight 会增加开销；其采集目录应单独标识，不使用受干扰的记录作正式性能结论。`profile` 入口自动标记所有记录并导出 PyTorch `trace.json`。不安装 Nsight 也能完成实验主体。

## 适配依据

固定版本的 [Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py) 将显式 `position_ids` 用于 RoPE，并将 K/V 交给 Cache 更新。[DynamicLayer 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py) 在 seq 维追加张量。本平台基于该行为分离逻辑 RoPE 位置与物理 mask/存储位置，并用真实 Qwen3 tiny 模型的 dense mask oracle 验证多步结果。未将这套推断泛化到 StaticCache、滑窗、动态 RoPE 或其他架构。
