# KVLab：推理状态管理与数据访问优化

可实际运行的单卡、batch=1、冻结权重、纯文本稠密 Qwen3 实验平台。CLI 贯通 **full KV → prompt KV 物理干预 → 固定轨迹/自由生成质量评测 → 计时/显存 → 静态分析报告**。默认模型 `Qwen/Qwen3-8B`，支持本地路径、独立 tokenizer 和 revision。

第一版适配 **PyTorch 2.8.0、Transformers 4.57.1 的 Qwen3 + 全注意力 DynamicCache/DynamicLayer**。这是有意限定的可验证实现，不承诺所有 Transformers 模型通用兼容。小模型只验证实现正确性，不能据此推断 8B 性能。

## 安装与最短路径

从仓库根目录执行，服务器目标版本为 **Python 3.10.12**（项目最低版本也是 3.10.12），建议 Linux，CUDA 与 PyTorch 2.8 对应。CUDA wheel 按你的驱动环境从 PyTorch 官方源安装；下面安装会复用已安装的同版本 torch。没有 GPU 也能运行 CPU tiny 测试。

```bash
python3.10 -c 'import sys; assert sys.version_info[:3] == (3, 10, 12), sys.version'
python3.10 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -e '.[test]'
python -m pip check
python -m kvlab env --config configs/environment.yaml --output outputs/environment.json
python -m pytest -q
```

虚拟环境沿用创建它的解释器；已有其他 Python 版本创建的环境应换用新目录，例如 `.venv310`，再激活对应环境。依赖固定为 PyTorch 2.8.0、Transformers 4.57.1、NumPy 2.2.6、matplotlib 3.10.7 和 PyYAML 6.0.3。[NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/) 支持 Python 3.10，原先的 2.3.4 要求 Python >=3.11；模型与 Cache 相关依赖版本保持原值。新旧依赖环境的实验应分别保存环境元数据，避免将不同版本的结果无条件合并。

`env` 不加载/下载模型；依赖未安装时 `python -m kvlab env` 仍可检查缺失项。指定 CUDA 配置但无 CUDA 时返回退出码 2。

GPU 机器上安装并激活 Python 环境后，首先执行这三条命令：

```bash
python -m pip install -e '.[test]'
python -m kvlab env --config configs/environment.yaml --output outputs/environment.json
python -m kvlab run --config configs/gpu-smoke.yaml
```

第三条使用仓库内的小数据文件，执行 full/no-cache 前检、三种策略自由生成和 teacher forcing，自动输出报告目录。它验证目标模型链路，不代表完成性能扫描。也可执行 `bash scripts/gpu_smoke.sh`。

无 GPU 的完整离线流程：

```bash
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny.yaml
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-diagnostic.yaml
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-performance.yaml
```

`tiny` 是本地随机初始化的真实 `Qwen3ForCausalLM`：2 层、4 query heads、2 KV heads（GQA）、RoPE，搭配测试用 byte tokenizer，无需下载权重。CPU 时间写入 `*_cpu_ms`，GPU 字段保持 `null`；CPU 性能流程不输出加速比。

## 正式实验

先生成固定数据，再运行。默认 64 条实体事实约可放入 2K Qwen tokens；最终长度必须以实际 tokenizer 的预检为准，过长会明确失败。

```bash
python -m kvlab synth --output data/synthetic.jsonl --count 8 --records 64 --seed 42
python -m kvlab run --config configs/baseline.yaml
python -m kvlab run --config configs/quality.yaml
python -m kvlab run --config configs/quality-budgets.yaml
python -m kvlab run --config configs/strategies.yaml
python -m kvlab run --config configs/diagnostic.yaml
```

| 配置 | 用途 |
|---|---|
| `environment.yaml` | 仅环境/配置检查，默认目标 CUDA |
| `gpu-smoke.yaml` | Qwen3-8B 256 输入、16 生成的三策略质量 smoke |
| `baseline.yaml` | full，2K/8K/16K/24K 输入，固定生成 256，预热 2、重复 5 |
| `strategies.yaml` | 相同输入扫描 full/recent/random_recent，多预算 |
| `quality.yaml` / `quality-budgets.yaml` | 正常 EOS；独立自由生成 + 同轨迹 teacher forcing |
| `diagnostic.yaml` | 小规模选定层/head/query 的分块 attention 重算 |
| `tiny*.yaml` | 无权重下载的 CPU 正确性、诊断、固定步数流程验证 |

CLI 可覆盖任意已知键，未知键会失败；列表整体替换。配置支持相对本配置文件的 `extends`，数据/输出路径相对于执行目录。

```bash
python -m kvlab run --config configs/quality.yaml \
  --set model.name=/models/Qwen3-8B \
  --set model.local_files_only=true \
  --set model.revision=main \
  --set 'data.ids=[s0000,s0001]' \
  --set 'data.input_lengths=[2048]' \
  --set generation.mode=sample \
  --set generation.temperature=0.7 \
  --set generation.top_p=0.9 \
  --set seed=123 \
  --set label=local-quality
```

模型与 tokenizer 分别支持 `name/tokenizer`、`revision/tokenizer_revision`。正式远端实验建议固定不可变 commit；请求 revision 和解析出的 `_commit_hash` 保存到结果。权重不随结果复制，本地目录应由研究者版本化且保持只读。`dtype` 支持 float32/float16/bfloat16；`device` 支持 cpu/cuda:N。`data.chat_template`、自定义 Jinja `data.template` 与 `data.thinking` 均可配置，实际模板保存到 `model.json`。Qwen thinking 关闭仍需遵循其 tokenizer 模板；开启 thinking 的任务评分只看闭合 `</think>` 之后的答案。

输入扫描会**仅在整个模板之前添加明确记录的无关 token 前缀**，达到精确 token 数；保留原始任务、完整模板及证据。原始 prompt + 完整 token IDs、填充位置/数量都保存。该构造用于受控长度实验，会改变位置和 prompt 分布；并非真实长文档基准。任务本身长于目标时拒绝截断；输入 + 最大输出超过模型 `max_position_embeddings` 时，在测量前全量预检失败。不会修改 RoPE 配置。已有静态 linear/YaRN 配置可用；动态 RoPE、滑窗、混合 Cache 被拒绝。

## KV 策略与位置语义

| 策略 | prompt KV 保留规则 |
|---|---|
| `full` | 原 Cache 原样使用，无收集、无额外固定位置 |
| `recent` | 最近 `min(budget,prompt_length)` 个 prompt 位置 |
| `random_recent` | 在相同预算内先保留最近 `recent_window`，再从其余 prompt 位置无放回随机选择 |

所有层和 KV heads 使用共同索引，索引按原始位置升序。随机选择使用独立 Python RNG，样本/重复 seed 保存；同输入、同 seed、同策略可重现。预算大于输入时封顶，实际数量、固定窗口和额外位置（本版为空）显式保存。

策略先统一补齐省略的 `budget: null`（仅 full）和 `recent_window: 0`，再按同一规则生成结果 ID 并检查重复。省略字段与显式默认值视为同一策略，重复配置在加载模型前报错。`full/recent` 的 `recent_window` 只能省略或为 0，非零值会报错；`random_recent` 的窗口必须为预算内的非负整数。合法配置的 ID 保持为 `full`、`recent-b预算-w0`、`random_recent-b预算-w窗口`。

**只在完整 prefill 后干预一次。预算是保留的 prompt KV 数，不是全程总 KV 上限。** 随后新生成 KV 正常追加；生成 N 个输出 token 需要 N−1 次 decode 前向，因此最终 Cache 通常包含 `保留 prompt 数 + N−1` 个位置，最后一个输出 token 尚未送回模型。首个输出 token 使用完整 prefill 的 logits，干预在采样前执行，因此第一步局部误差为零，但策略成本包含在 TTFT。

`position_ids=P+t` 使用原始绝对序列位置，为新 query/key 计算 RoPE。`cache_position=当前物理长度` 只负责紧凑存储和 causal mask。已存 K 保留原始 RoPE 旋转，收集不会重旋转。由于 batch=1、无 padding、每次只解码一个 token，所有保留历史与当前 token 均合法可见。禁止将压缩长度用于后续 RoPE 位置。物理张量减少、后续 dense attention 的 key 长度减少；没有实现稀疏 kernel、量化或卸载。

每个方法、每个重复都独立重新 prefill；不复用被裁剪的 Cache。运行前用短前缀执行 full/no-op Cache 与无 Cache 连续前向检查，并保存误差和容差口径；测试另与 HF `generate`、稠密 mask oracle 多步对照。短前检不能代替大模型长上下文实测。

## 两种采集路径与质量评测

行为模式 `mode: diagnostic` 安装 hook，仅重算选定 decode query 的 QK 分布：支持 layers/heads/steps、key 数上限、记录数上限、分块大小、直方图和 top-k。使用模型真实 Q 归一化、已旋转 K 与 GQA 映射。两遍 FP32 分块 logsumexp/聚合，不保存全层、全步或 prefill 的 N×N attention。原 forward backend 与重算方法明确记录；采集上限跳过原因保存在 `diagnostic_skips`。默认 SDPA 不需要为了 attention 切换 eager；可显式改 eager 做小输入诊断。诊断耗时不能用于正式性能结论。

正式性能 `mode: performance` 关闭诊断 hook、行为导出、逐步日志和 profiler，要求 SDPA；CUDA 正式性能只允许 `flash/efficient`，不可用就失败，不静默换 backend。`math` 保留用于质量/诊断对照与 CPU 流程检查，结果保留其真实实现标识。固定步数必须显式 `ignore_eos: true`。冻结权重、inference mode、未使用 compile/CUDA Graph，因此无首次编译成本。

质量 `mode: quality` 使用正常 EOS，保存完整输出、token IDs、实际长度、终止原因、到达长度上限标记和截断标记。EOS 恰好出现在上限位置时，达到上限但不算截断；固定步数性能运行的截断率不适用。每个重复先从 full 自由输出构造参考 continuation，各策略（包括 full）用独立 Cache 在同一个参考 token 序列上 teacher forcing，保存轨迹 hash 和长度。逐位置 FP32 `log_softmax`，计算 `KL(full || strategy)`、logits RMSE、分布 L1，选定位置保存双方 top-k 分布；所有位置计算平均 NLL、相对 full 的 NLL 变化和平均 KL。logits 和统计检查 NaN/Inf，不使用任意 epsilon 概率截断。参考包含自然 EOS token（若出现）。局部误差不能替代自由生成质量。

JSONL 数据必须含字符串 `id`、`prompt`，可选 `reference`（字符串或多答案列表）、`task`、`metadata` 及其他任务字段，原样保存到 samples。默认 exact match 忽略大小写和多余空白；同时给出词边界 answer-contains，不能将其当成更严格准确率。缺 reference 时分数为 null。合成任务含单条实体检索和三跳关联检索，保存构造规则、随机种子、打乱后的证据记录索引及原 prompt 字符位置；字符位置不是 token 位置，实际输入以持久化 token IDs 为准。合成检索不等于普遍推理能力。`data/real-example.jsonl` 是外部数据格式接入示例，需替换为有权使用的真实数据；未伪称已经运行真实 benchmark。不依赖付费 API 或 LLM judge。

## 计时、显存与报告

详细定义见 [测量说明](docs/measurement.md)。每次运行产生 `outputs/<UTC时间>-<label>/`：

```text
config.source.json / config.resolved.json   配置源、覆盖与完整有效配置
environment.json / model.json / code.json  依赖、GPU、revision、模板、源代码 hash
samples.json / full_validation.json        原始样本与运行前 full 正确性预检
records.jsonl / records.csv                所有测量重复、输出、位置、质量、显存与逐步时间
summary.json / summary.csv                 按模式、长度、任务、策略聚合，保留 n 与离散程度
comparisons.json / comparisons.csv         同次运行内兼容记录与 full 的配对比较
status.json                               完成/失败、错误、警告；失败也尽量生成报告
01-context.png ... 06-relative.png         六类真实结果图，缺失数据标记 Not collected
report.html                               可直接离线打开；无服务、无 CDN
trace.json                                仅独立 profile 命令生成
```

直接用浏览器打开 `report.html`。可从原始记录重建：

```bash
python -m kvlab report outputs/<实际运行目录>
```

报告自动展示上下文/decode/KV、预算/质量、预算/请求时间、模型/管理成本、历史访问、相对 full 的 NLL 变化与加速比。PNG 使用英文轴标签保证无中文字体机器可读，HTML 的解释为中文。缺数据保持 null 并标记未采集，不填零。为避免错误合并，第一版**每次运行独立报告，不跨运行自动比较**；仅相同模式、输入 hash、sample、seed、repeat 与 full 配对，baseline 和单位/n 明示。性能与质量结果分别展示，无法自动组合为同一速度质量点。

预算—完整请求耗时图保留完整策略 ID 和输入长度。同预算不同 `recent_window`，或预算封顶后实际保留数相同的不同请求预算，会显示为独立系列；中位数仅在同一系列内聚合。已有运行可用上面的 `report` 命令重新生成图表，无需重新执行模型。

Profiler 独立执行：

```bash
python -m kvlab profile --config configs/gpu-smoke.yaml --set measurement.warmup=1
# 无 GPU 也可检查 CPU trace：
OMP_NUM_THREADS=1 python -m kvlab profile --config configs/tiny-performance.yaml --set measurement.repeats=1
```

将 `trace.json` 导入 Perfetto 或 Chrome tracing。独立入口记录 shapes、allocator memory，不用于正式性能结论。可选 Nsight 命令见测量说明。

## 工程入口与扩展约束

| 文件 | 职责 |
|---|---|
| `kvlab/adapter.py` | 模型加载、backend 检查、KV 物理收集、RoPE/位置语义 |
| `kvlab/strategies.py` | 策略选择，预算、索引、固定位置记录 |
| `kvlab/runner.py` | 独立请求、预热/重复、固定参考轨迹、结果持久化 |
| `kvlab/timing.py` | CUDA Events、阶段显存、采样 |
| `kvlab/diagnostics.py` | 有上限的选定 query attention 分块重算 |
| `kvlab/data.py` / `metrics.py` | JSONL、可复现合成数据、质量指标 |
| `kvlab/report.py` | 兼容性配对、聚合、CSV/HTML/PNG |
| `kvlab/config.py` / `environment.py` | 严格配置与可追踪环境/代码标识 |
| `tests/` | 位置/Cache/指标/报告等高风险正确性测试 |

新增选择策略时在 `strategies.select` 实现并扩展 `strategies.normalize_strategy/strategy_id` 的参数与身份规则：返回升序、唯一、原始 prompt 索引；明确 budget、实际保留与任何固定额外位置，使用独立可重现 RNG。attention-based 策略可在 runner 的干预阶段获取预先定义的统计，但统计采集和选择成本必须进入请求/TTFT，正式性能路径不能借用诊断运行的免费信息。

量化、CPU 卸载和稀疏 kernel 的扩展入口是 adapter 的 `new_cache/intervene/decode/cache_bytes/theory_bytes`；这些未实现功能当前明确拒绝，不能只注册名字后沿用不兼容 Cache。新适配器须保留逻辑/物理位置分离、GQA 与 mask 语义，不能改权重或污染其他请求；测试必须包含 full/no-op 一致性、mask oracle、多步位置和独立状态。更新 Transformers 版本需重做上述验证。已有论文算法应另命名实现并解释一致性和偏差，不能把这里的简单策略称为 SnapKV/H2O。

测量扩展必须保存所有重复、纳入采样/选择/搬运成本、限制 backend fallback、隔离诊断与 profiler，并记录是否只是 mask 选择或实际减少物理张量/计算。不能将理论 KV 字节当成 HBM 流量；kernel 与 allocator 的真实性能收益须在目标 GPU 上实测。

本机实际验证与尚未验证范围见 [验证记录](docs/validation.md)。
