# 本机验证记录与限制

## 按实际数据生成产物（2026-11-03）

结果生成规则已调整：只有成功完成且存在正式记录时才导出对应 CSV、summary/comparisons JSON 和有数据的 PNG；空配对表不导出。图中不再包含 `Not collected` 占位子图。失败运行保留状态、错误、配置、实际采集的诊断及已完成请求的原始 JSONL，不将部分记录聚合为结果。不完整的参考轨迹保留在 `full_validation.json` 内，不另写 `validation_trajectory.json`。重建报告只清理固定命名的派生图表，保留原始记录、诊断和其他用户文件。

Python **3.10.12** 全套测试 **122 passed、1 skipped（CUDA）**，日志：`outputs/conditional-artifacts-verification/pytest.txt`。新增回归覆盖空运行、独立预检、预检失败、生成中途失败、模式变化后的旧文件清理、无标注与无配对、实际零值保留、单个子图布局、缺失成本项、diagnostic 采集缺失及 CPU/profiler 的性能图隔离。原先强制生成六图的旧测试已更新为按实际数据检查。

CPU tiny 实际 CLI 及文件清单记录在 `outputs/conditional-artifacts-verification/cli-audit.json`：quality 仅生成 01/02/06，diagnostic 仅生成 01/05，CPU performance 仅生成 01，独立 validate 和超长输入失败不生成结果图或派生表。每个目录又通过 `report` 重建，核对旧占位图被清理、原始 JSONL 字节保持不变。已目视检查 quality 的单 KV 子图和单 NLL 子图，无空白半图。未在本机重新执行 CUDA 性能测量；CUDA 绘图筛选通过受控 fixture 验证，不作为真实 GPU 性能结果。

以下历史记录中的“每次生成六张图”和 `Not collected` 为旧版行为，新运行及显式重建报告采用上述规则。

实施日期：2026-10-21（Asia/Shanghai）。初始目录为空，无已有代码或适用 AGENTS.md。主机为 macOS ARM，无 NVIDIA GPU、无目标模型。最初未安装第三方依赖，已在工作目录 `.venv` 安装固定版本并以 editable 方式安装本项目。

## 实际执行的检查

```bash
python -m pip install -e '.[test]'
python -m compileall -q kvlab
python -m kvlab env --config configs/environment.yaml --output outputs/environment.json
OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny.yaml
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-diagnostic.yaml
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-performance.yaml
OMP_NUM_THREADS=1 python -m kvlab profile --config configs/tiny-performance.yaml \
  --set measurement.repeats=1 --set 'data.input_lengths=[512]'
python -m kvlab synth --output data/synthetic.jsonl --count 8 --records 64 --seed 42
```

- 环境：Python 3.13.2、PyTorch 2.8.0、Transformers 4.57.1、NumPy 2.3.4、matplotlib 3.10.7、PyYAML 6.0.3、pytest 8.4.2。完整依赖版本见每次运行的 environment.json。
- 环境检查正确返回 `cuda_available=false`，指定 GPU 配置时退出码 2；这是预期结果。
- 正确性测试覆盖 full 与无 Cache/原始 HF generate、一致预算与随机性、GQA/RoPE 多步紧凑 Cache 对 dense mask oracle、Cache 隔离、固定参考轨迹输入、KL/NLL 数值方向与 NaN/Inf 拒绝、EOS/固定步数、输入不截断、显式失败、本地模型/tokenizer 载入、诊断分块对 eager 真 attention、采集上限与 hook 清理、聚合与缺失值、完整产物和失败报告。
- CUDA 测试在本机明确跳过；它会在具有支持 Flash BF16 的 CUDA GPU 上检查 Events、显存与物理 KV，并运行 full/no-cache 预检。
- tiny 质量、诊断、固定步数流程均已实际跑通；输出目录中的数据来自本地随机 Qwen3 的真实执行。每份报告包含六张 PNG、原始 JSONL、CSV、模型输出和可离线打开的 HTML。已目视检查历史访问热图；缺少 CUDA 性能数据的图明确显示 Not collected，没有补造 GPU 数字。
- 独立 CPU Profiler 已导出可解析的 trace.json；所有该类记录标记 profile，报告不计算其加速比。
- 本机环境预置的 `HUGGINGFACE_CO_RESOLVE_ENDPOINT` 会触发 Transformers 的弃用提醒；未修改用户环境变量。初次 matplotlib 字体缓存可能较慢，报告逻辑已将缓存定向到可写输出目录。两者均在模型测量区间之外。

最终自动测试数量和实际产物审计保存在 `outputs/verification.json`；最终测试输出保存在 `outputs/pytest.txt`。这些文件记录本机检查，不是目标模型性能结果。

## 尚未验证

1. **Qwen3-8B 真实权重、真实 tokenizer、真实 chat/thinking 输出质量**：本机未下载目标模型。已测试同一 Qwen3 实现的随机 tiny 权重，并测试真实 AutoModel/AutoTokenizer 本地保存载入路径。
2. **CUDA Events、Flash/Efficient kernel、allocated/reserved 阶段显存与 2K–24K 性能**：本机无 CUDA。已提供目标模型 gpu-smoke 与完整扫描配置，必须在目标 GPU 执行后才有结论。
3. **已有 linear/YaRN 配置**：配置允许静态类型，但未在本机验证目标长上下文扩展。程序不会添加/修改扩展配置。动态 RoPE 和滑窗拒绝执行。
4. **不同 GPU、驱动、kernel 的跨平台 bitwise 重现**：保存 seeds 和依赖/设备，但未强制确定性 kernel，以免改变默认性能路径；结果可能存在正常数值差异。
5. **真实数据集与任务专用指标**：提供 JSONL 接入和无标注行为，默认 exact match/answer-contains；没有声称合成检索代表普遍推理能力。

## 有意保留的第一版限制

- 单设备、batch=1、一次完整 prefill、随后单 token decode；不支持 padded batch、beam search、滑窗、混合架构、静态/分页 Cache、训练、动态总 KV 预算。
- 首 token 使用完整 prefill logits，干预不能改变它；其策略耗时计入 TTFT。保留 prompt KV 的表示本身包含完整 prefill 的历史信息，因此不等价于从输入中删除对应 token 后重新 prefill。
- attention 诊断是选定 query 的 FP32 分块重算，存在与原半精度 kernel 的正常舍入差异；当前不提供 attention 输出相对误差。
- 管理时间包含索引计算/收集，尚未细分每层 gather kernel；策略临时显存峰值未单列，prefill/decode 峰值已记录。
- 请求 wall 定义为离线内存 prompt 到输出文本，不是在线排队/网络请求；模型加载、数据文件读取和报告不在边界内。
- 本地模型记录小文件 SHA 和大文件大小/mtime，不重新扫描全部大权重生成 SHA；研究者需固定本地权重版本。远端请求及实际模型/tokenizer commit 会保存。
- 报告只在同次运行内自动配对，不跨运行混合不同配置。质量与性能模式独立，速度质量图有缺失项时明确未采集。
- 没有实现量化、CPU 卸载、attention-based selection、稀疏 kernel、vLLM/SGLang。扩展入口已保留且当前不支持的配置明确失败。

## 目标 GPU 验收顺序

先按 README 的三条命令完成安装、环境检查和 gpu-smoke，再执行 `pytest -q`，检查 CUDA 测试和 full_validation.json。确认 backend 为指定的 SDPA kernel、模型 frozen、首 token 来源、逻辑与物理位置映射和实测 KV 字节均正常，再生成数据，依次执行 baseline、quality、strategies 和 diagnostic。性能图只读取 performance 模式的真实 CUDA 样本；质量比较必须保留正常 EOS 和截断率。

## Python 3.10.12 兼容性复核（2026-10-23）

服务器目标版本为 Python 3.10.12。项目最低版本改为 `>=3.10.12`，NumPy 固定为 2.2.6；PyTorch 2.8.0、Transformers 4.57.1、matplotlib 3.10.7、PyYAML 6.0.3 和 pytest 8.4.2 保持原值。前文的 Python 3.13.2 / NumPy 2.3.4 是首次验证的历史环境。

本次在 macOS ARM 上，从 Python 官方 3.10.12 源码构建独立解释器，在临时目录创建新虚拟环境，使用对应 Python 3.10 的 wheel 安装项目及测试依赖。实际 `sys.version_info[:3]` 为 `(3, 10, 12)`；完整依赖版本保存在 `outputs/python31012-validation/requirements.txt`，安装输出保存在同目录 `install.txt`。没有用其他 3.10 补丁版本代替本次运行验证。

激活该独立环境后执行：

```bash
python -c 'import sys; assert sys.version_info[:3] == (3, 10, 12), sys.version'
python -m pip check
python -m compileall -q kvlab
OMP_NUM_THREADS=1 python -m kvlab env --config configs/tiny.yaml \
  --output outputs/python31012-validation/environment.json
OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny.yaml --set label=python31012-quality
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-diagnostic.yaml --set label=python31012-diagnostic
OMP_NUM_THREADS=1 python -m kvlab run --config configs/tiny-performance.yaml --set label=python31012-performance
```

- `pip check`：`No broken requirements found.`；代码编译检查通过。
- 完整测试：**47 passed, 1 skipped, 1 warning**。跳过项为 CUDA 测试；warning 为本机已有的 `HUGGINGFACE_CO_RESOLVE_ENDPOINT` 弃用提醒。原始输出见 `outputs/python31012-validation/pytest.txt`。
- 三个 tiny 流程均为 `complete`，产物的环境记录均为 Python 3.10.12 / NumPy 2.2.6。quality 生成 12 条自由生成记录和 12 条 teacher-forcing 记录；diagnostic 生成 6 条自由生成记录和 72 条 attention 诊断记录；performance 生成 12 条固定 8 token 记录。
- 已直接核对原始记录：同一配对组的参考轨迹 hash 一致，NLL/KL 为有限值，full 的 NLL 差和 KL 均小于 `1e-6`；三个流程的短前缀 full/no-cache 预检均通过，最大 logits 绝对误差为 `1.1920928955078125e-07`。每份报告包含六张非空 PNG 和非空 HTML，decode 次数为生成 token 数减一。CPU 记录的 GPU 时间保持 null，`gpu_measurement_valid=false`。
- 产物目录分别为 `outputs/20261023T084108.419623Z-python31012-quality`、`outputs/20261023T084128.896154Z-python31012-diagnostic`、`outputs/20261023T084148.754309Z-python31012-performance`。汇总核对结果见 `outputs/python31012-validation/audit.json`。

本次确认的是精确 Python 3.10.12 下的安装和 CPU tiny 执行路径。Linux 服务器的驱动、CUDA kernel、Qwen3-8B 真实权重及性能仍须按目标 GPU 验收顺序运行；上述结果不构成这些项目的验证。

## Flash 预检 mask 修复（2026-10-28）

服务器 L40 日志报告 `Flash Attention does not support non-null attn_mask`。`Adapter.validate_full` 的无 Cache 参考前向未传 padding mask，触发 Transformers 4.57.1 的 packed-sequence mask 构造分支，向 SDPA 传入显式四维因果 mask。CPU math 能处理这个 mask，严格指定的 CUDA Flash 路径会拒绝。

修复仅在该参考前向的模型入口传入二维全 1 boolean padding mask。对于这里的完整、无 padding、单序列输入，HF 将其优化为 SDPA 的 `attn_mask=None, is_causal=True`，保留因果约束。正式 prefill/decode 及严格 backend 选择保持原路径；预检在正式计时之前执行。

新增 `regression_tests/test_validation_mask.py`，使用真实 tiny Qwen3 和 math SDPA，检查最终 SDPA 边界的 mask 与 causal 标志，并确认三个无 Cache 参考长度均实际执行。覆盖单 token、普通长度和超过预检 32 token 前缀上限的输入。旧实现已在该测试的无 Cache oracle 处失败；修复后完整测试为 **50 passed, 1 skipped, 1 warning**，CUDA 测试因本机无 GPU 跳过，warning 为已有 HF 环境变量弃用提醒。

本轮实际运行环境为本机 Python 3.13.2 / CPU、PyTorch 2.8.0、Transformers 4.57.1；变更文件另通过 Python 3.10 语法检查。本轮未重新验证精确 Python 3.10.12 或 CUDA Flash，前一节的 3.10.12 结果属于修复前的历史记录。服务器继续使用 Python 3.10.12，在同步变更并激活其虚拟环境后执行：

```bash
python -m pytest -q regression_tests/test_validation_mask.py
python -m kvlab run --config configs/gpu-smoke.yaml
```

第二条保留原来的 BF16 / SDPA Flash 配置。需确认新运行 `status.json` 为 `complete`、`full_validation.json` 中 `passed=true`，且不再出现非空 mask 的拒绝警告；本机 CPU 结果不代表这些 CUDA 检查已经通过。

## 细化预检诊断与固定轨迹重放（2026-10-30）

新增 `kvlab/validation.py` 和独立 `python -m kvlab validate` 命令。预检比较 A（平台 cached full/no-op）、B（直接 HF cached，默认推导位置）、C（无 Cache 重算），先从 C 生成并保存参考 token，A/B 重放相同轨迹。默认仍取第一样本、第一输入长度的前 32 token，采集 prefill 和三个 decode 步。配置、采集范围、指标方向和重放命令见 README 的“预检诊断与固定轨迹重放”。

报告逐步保存实际输入、RoPE/物理位置、每层 KV、SDPA 最终参数及启用的 backend；额外验证 full/no-op 内容、地址和元数据不变、A/B Cache 不共享存储。数值统计在 CPU FP64 上计算，旧数值验收仍为 decode 步 A_vs_C 的 FP32 allclose，容差未扩大。prefill/B 对照及 KL、top-1 等只用于诊断。三态标志区分通过、已发现失败和未完成检查。普通数值超差会完成剩余步骤并保存报告，语义、非有限值或执行异常则保存部分报告；两类失败都阻止正式测量。观测 hook 和 SDPA 临时包装在退出时清理。

本轮重新从官方源码构建精确 Python **3.10.12**，在 `outputs/py31012-validation.u95nki/venv` 独立环境安装项目和固定依赖，实际使用 PyTorch 2.8.0、Transformers 4.57.1、NumPy 2.2.6；macOS ARM / CPU，无 CUDA。完整依赖见该目录的 `requirements.txt`，`pip check` 无依赖冲突，编译检查通过。最终完整测试为 **76 passed, 1 skipped, 1 warning**，原始输出在 `outputs/py31012-validation.u95nki/pytest-final.txt`。跳过 CUDA 测试；warning 为已有 HF 环境变量弃用提醒。

新增回归覆盖近零 logits 超差、共同偏移与分布不变、NaN/Inf JSON 安全、三路径实际位置/KV/SDPA 参数、重放期间禁止重新 argmax、输入/身份/hash/长度不匹配的提前拒绝、普通数值失败后继续采集、位置错误/错误 causal 标志/full 原地修改/非有限输出/kernel 异常的停止与清理，以及 runner 失败报告落盘、独立 validate 的退出码和禁止正式测量。

Python 3.10.12 下实际执行了以下 CLI 链路：

- `validate --config configs/tiny.yaml`：`outputs/20261030T053127.773372Z-validation-detail-fp32`。
- `validate --config configs/tiny.yaml --set model.dtype=bfloat16` 并重放上一份轨迹：`outputs/20261030T053147.508554Z-validation-detail-bf16-replay`。
- `run --config configs/tiny.yaml`，单样本、三 token、无预热、一次重复：`outputs/20261030T053206.704946Z-validation-detail-quality`。

三个运行均完成四个预检步骤、生成六张 PNG 和 HTML；前两个没有正式生成记录，第三个生成三条自由生成和三条 teacher-forcing 记录，GPU 有效标志均为 false。两种精度的参考 token 和轨迹 hash 一致。CLI 产物核对结果见 `outputs/py31012-validation.u95nki/e2e-audit.json`。

这些结果仅验证 CPU tiny 的实现路径。服务器 L40 / Qwen3-8B 的 BF16、FP16、FP32 差异尚未重新采集；诊断增强不会自动将它们判为正常或放行。应在服务器 Python 3.10.12 中用 validate 保存完整报告与同轨迹对照后，再分析或校准数值验收标准。
# 分层数值判定验证（2026-11-02）

预检 schema=3 使用 `layered-v1`：所有 prefill/decode 步必须完成结构与有限性检查；A/B cached allclose、三组比较的 KL/概率 L1/参考 token NLL 绝对差以及 top-1 为硬门槛。A/C、B/C 原始 logits 超差和全词表最大 log-probability 差单独告警。`pass_with_warnings` 允许继续运行，JSON、status、终端和 HTML 保留告警。阈值按实际模型 dtype 选择，完整配置及生效策略分别持久化；默认值是试运行候选，不是已校准的跨模型/backend 标准。具体数值和覆盖方式见 README 的预检章节。

使用项目内独立 Python **3.10.12** 环境 `outputs/py31012-validation.u95nki/venv/bin/python`，全套测试 **107 passed、1 skipped（CUDA）**；现有 HF 环境变量弃用警告不影响结果。新增测试覆盖每个 dtype/指标阈值边界、top-1 近并列变化、共同 logits 偏移、低概率目标 NLL、大幅尾部 log-probability 告警、平台 prefill 故障、两条缓存路径共同出错、未完成检查、非法配置与告警状态持久化。原有位置/mask/KV/NaN/kernel 故障测试继续通过。日志：`outputs/layered-validation-verification/pytest.txt`。

真实 CPU tiny CLI 在 float32、float16、bfloat16 下分别完成独立 validate（schema=3、没有正式测量记录），正常 quality 流程产生 3 条自由生成与 3 条固定轨迹记录。审计：`outputs/layered-validation-verification/cli-audit.json`。受控共同偏移测试验证了带告警通过的退出/报告行为；这些故障数据不作为 GPU 数值结果。

只读分析用户提供的 `bf16_full_validation.json` 和更新后的 `fp16_full_validation.json`，确认实际 SDPA 张量精度分别为 bfloat16、float16，启用 math backend。两份均为同一 FP32 token 轨迹重放，各步 A/B logits 完全一致。BF16 的最大 KL、概率 L1、目标 NLL 差异分别为约 1.588e-4、0.002360、0.001190；FP16 分别约为 3.182e-6、0.0004811、0.0002401。根据已保存指标，两份均满足默认分层硬门槛，仍保留 logits 偏差告警。两份旧报告没有完整词表最大 log-probability 差，不能从保存的少量误差分量重建其最大值；保存分量给出的下界分别约为 8.539 和 0.8408，已足以提示尾部偏差。离线审计记录来源 SHA256、阈值和逐步判定：`outputs/layered-validation-verification/server-report-audit.json`。原始报告未被改写，旧的 passed=false 保持不变。

未在本机重新执行 L40/Qwen3-8B，不声称完整新报告、Flash backend、长上下文或真实任务已在服务器验证。当前服务器证据只有短重复前缀和 3 个 decode 步，且 top-1 概率较高；正式阈值仍须在代表性输入、不同长度与受控错误上校准并进行独立验证。

