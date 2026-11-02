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

## 预检诊断与固定轨迹重放

`run` 和 `profile` 在任何预热、计时和 profiler 之前执行预检。新增 `validate` 入口只采集预检并生成报告，不进入正式生成或测量：

```bash
python -m kvlab validate --config configs/gpu-smoke.yaml \
  --set model.dtype=float32 --set model.sdpa_kernel=math \
  --set label=precheck-fp32-math
```

每次保存 `full_validation.json`、`validation_trajectory.json` 和 HTML 中的逐步诊断表。数值硬门槛失败后继续采集剩余计划步骤，最终返回失败退出码并阻止正式测量；位置、输入、Cache、mask、非有限值或 kernel 异常则保存已有证据后提前停止。单独的数值告警不阻止测量，但会在终端、status 和 HTML 中显示。即使上面命令因数值超差退出，完整生成的参考轨迹仍可重放；`complete=false` 的不完整轨迹会被拒绝。

报告的检查标志使用三态：已发现失败为 `false`，完整执行并通过为 `true`，提前中止且尚未发现该类错误为 `null`。必须同时查看 `completed_steps/planned_steps` 和 `fatal_error`，不能将未完成的检查视为通过。

默认 `validation.prefix_tokens=32`、`validation.decode_steps=3`、`validation.top_k=8`、`validation.trajectory_path=null`。前缀是实际准备后输入的最前面部分，可能主要是填充文本；可显式设 `validation.prefix_tokens=256` 检查 gpu-smoke 的整个输入。仅检查第一个样本和第一个输入长度，不代表整个数据集或长上下文已验收。

三条路径为 A：平台 full/no-op cached；B：直接 HF cached，由 HF 推导位置；C：无 Cache 重算完整前缀。A/B 使用独立 Cache，full/no-op 额外核对 KV 内容、地址和元数据。C 路径先生成参考 token，A/B 始终喂入相同 token；固定步数忽略 EOS，仅用于数值预检。step=0 比较 prefill 对 y0 的预测；step=1..3 分别喂入 y0..y2，比较对 y1..y3 的预测，最后一个参考 token 不送回模型。

逐步报告包含三组比较 A_vs_C、A_vs_B、B_vs_C，前者为 candidate、后者为 reference，例如 A_vs_C 的 KL 方向是 C‖A。指标包括 max/mean absolute error、RMSE、共同偏移和 centered RMSE、违规数量/比例、违规分量的双方 logits/允许误差、signed KL(reference‖candidate)、概率 L1、目标 NLL 及其绝对差、全词表最大 log-probability 差、top-1 概率与 top-1/top-2 logit 间隔。KL 不夹为零；统计在 CPU FP64 上计算，不代表前向改用了 FP64，也不能恢复低精度前向的误差。每层保存实际输入、RoPE position_ids、物理 cache_position、KV 形状/dtype/device，以及 SDPA 边界的 Q/K/V、mask、is_causal、GQA 与启用的 backend。完整张量和 attention 矩阵不落盘。

预检报告 schema=3 使用 `layered-v1` 分层验收，覆盖 prefill 和所有 decode 步，不在步骤间取平均来掩盖单步失败：

- 结构与执行：语义、有限性检查全部通过，且所有计划步骤完整完成。
- 平台一致性：A_vs_B 必须通过 FP32 `allclose`，使用当前 dtype 的 `cached_atol/cached_rtol`。
- 分布近似：三组比较都必须满足 `abs(KL) <= kl`、概率 L1 `<= probability_l1`、当前固定参考 token 的 NLL 绝对差 `<= nll_abs`。KL 保留原始带符号统计，按绝对值判定，避免负数值噪声绕过门槛。
- 贪心选择：三组比较的 top-1 必须相同。即使 top-1/top-2 接近并列，也采用保守失败规则，双方 margin 保留用于定位。
- 数值告警：A_vs_C/B_vs_C 的原始 logits allclose 超差，以及任意比较的全词表最大 log-probability 差超出 `warn_logprob_max_abs`，只告警，不等同于分布验收失败。

默认阈值按**实际模型 dtype** 选择，所有数值为待校准的试运行值，未声称适用于全部模型、kernel 或长度：

| dtype | cached_atol | cached_rtol | kl | probability_l1 | nll_abs | warn_logprob_max_abs |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 2e-5 | 0.02 | 1e-7 | 1e-4 | 1e-3 | 1e-3 |
| float16 | 0.01 | 0.02 | 1e-5 | 0.002 | 0.001 | 0.01 |
| bfloat16 | 0.08 | 0.02 | 1e-3 | 0.02 | 0.01 | 0.1 |

通过 YAML 的 `validation.thresholds.<dtype>.<指标>` 或命令行覆盖，例如 `--set validation.thresholds.bfloat16.kl=0.0005`。阈值必须为有限非负数，不接受 bool、null、NaN/Inf 或未知字段。全部 dtype 的配置保存在 `config.resolved.json`，当前生效阈值另存于预检报告 `policy`。

报告另存 `implementation_checks_passed`、`distribution_checks_passed`、`top1_checks_passed` 及逐步 `assessments`。`numeric_status=pass` 表示硬门槛通过且没有告警，`pass_with_warnings` 表示硬门槛通过但仍有数值告警（退出码 0，允许正式运行），`failed` 表示硬门槛失败（退出码 1）。原有 `legacy_allclose_passed` 仍记录 decode 步 A_vs_C 的 allclose 汇总，可能为 false 而分层验收通过。告警不能被解读为全词表数值一致或低概率 token 的 NLL 已验证。

先用代表性任务、较不确定的分布和多个上下文长度校准，再在独立样本上验证；同时注入位置、mask、KV 更新错误，确认阈值不会放过实现错误。不同 dtype/backend 分别检查，不能为了让单个 smoke 样本通过而不断扩大阈值。所有观测 hook 和临时 SDPA 包装在结束/异常时恢复，预检需串行执行，不用于性能结论。

跨精度或 backend 比较时，显式重放第一份轨迹，避免各次运行重新选择不同 token：

```bash
python -m kvlab validate --config configs/gpu-smoke.yaml \
  --set model.dtype=bfloat16 --set model.sdpa_kernel=flash \
  --set 'validation.trajectory_path=outputs/替换为FP32运行目录/validation_trajectory.json' \
  --set label=precheck-bf16-flash-replay
```

重放检查前缀 token IDs、模型/tokenizer 身份、轨迹长度和 hash，允许 dtype/backend 改变。需保持同一前缀长度、步数、tokenizer 和模型权重；身份记录不是全部权重张量的校验和，本地权重仍须保持只读，远端建议固定 commit。重放仅固定预检输入，正式质量评测仍按自己的 full continuation 进行。

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
validation_trajectory.json                 预检固定参考 token、身份与 hash；可跨精度/backend 重放
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
