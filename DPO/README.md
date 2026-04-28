# DPO

## 1. 文档定位

本文档是 `DPO/` 模块的主文档，同时承担以下三种角色：

- Spec：需求规格说明
- Technical Design：技术实现设计
- Project Plan：开发排期与进度管理

`DPO/SPECS/` 下的分期文档用于承载每个阶段的详细设计。本文档负责定义跨阶段总目标、数据契约、目录约定、训练与评估口径、模块边界和项目计划。后续所有 DPO 需求变更，必须先更新本文档；若变更影响具体阶段，再同步更新对应 Phase SPEC。

DPO 模块承接总 `SPEC.md` 中的 DPO 设计，目标是在 SFT 模型已经学会基础输出格式和字段约束后，通过自动构建偏好对，让模型更偏向生成结构可用、代码可解析、可回测且 1 日 `|IC|` 更好的因子。

## 2. 模块目标

`DPO` 模块负责在 `SFT/` 已产出的训练数据、SFT adapter 和评估链路基础上，构建偏好数据并继续进行 Direct Preference Optimization。该阶段不是重新定义因子生成任务，而是在保持 SFT 输出契约稳定的前提下，把“更可用、更可执行、更有回测表现”的偏好注入模型。

核心目标包括：

- 提升结构化 JSON 输出成功率、字段合规率、Python 可解析率和回测通过率。
- 在通过硬约束的前提下，提升生成因子的 1 日 `|IC|` 表现。
- 保持与 SFT 完全一致的 assistant 输出契约，不新增输出字段。
- 复用 `extracter`、`SFT` 和 `backtest` 的已有校验能力，不引入独立 Eval 模块作为主要依赖。
- 保留 SFT adapter 作为只读基线，DPO 训练产物保存为新的 adapter。
- 在同一批 `val/test` prompt 上对比 SFT baseline 与 DPO model。

当前基线方案：

- 基础模型：沿用 SFT 的 `base_model_path`
- 初始策略模型：`base_model_path + sft_adapter_path`
- Reference model：冻结的 `base_model_path + sft_adapter_path`
- 微调方式：LoRA / QLoRA adapter 延续训练
- 偏好训练框架：优先采用 `trl.DPOTrainer`
- 偏好来源：同一 train prompt 下多个 SFT 候选的校验与回测结果

## 3. 范围定义

### 3.1 In Scope

- 基于 `SFT/data/train.jsonl` 生成 DPO 候选输出。
- 对候选进行结构化解析、字段校验、Python 校验和回测。
- 自动构建 `{prompt, chosen, rejected, metadata}` 格式的 DPO 偏好数据。
- 基于 SFT adapter 继续进行 DPO 训练，并输出新的 DPO adapter。
- 在同一批 `val/test` prompt 上对比 SFT baseline 与 DPO model。
- 产出候选生成记录、校验回测记录、偏好训练集、hard case、训练 manifest 和评估报告。
- 为后续 Agent 化因子研究系统保留稳定的模型加载、推理与评估接口。

### 3.2 Out of Scope

- 修改 SFT assistant 输出 JSON schema。
- 使用 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 构建 DPO 训练数据。
- 新建独立 Eval 模块替代现有 `extracter` / `SFT` / `backtest` 校验链路。
- 在 DPO 训练前强制 merge SFT adapter。
- 实盘策略组合、交易成本建模和多因子组合优化。
- 人工标注大规模偏好数据。
- 把 DPO 模型部署为线上服务。

## 4. 核心设计原则

- 训练数据防泄漏：DPO 偏好数据只来自 `SFT/data/train.jsonl`，`val/test` 只用于评估。
- 输出契约稳定：DPO chosen / rejected 均为 assistant JSON 字符串，字段仍限定为 `reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
- 自动偏好为主：偏好对默认由校验结果和 1 日 `|IC|` 自动生成，人工复核只用于抽查和 hard case 分析。
- 硬约束优先：非法 JSON、缺字段、代码不可解析、字段违规和无法回测是偏好排序前的主要过滤信号。
- adapter 不覆盖：SFT adapter 作为只读基线保留，DPO 训练输出保存为新的 DPO adapter。
- 默认不 merge：DPO 训练不要求先将 SFT adapter merge 到 base model；merge 只作为后续部署或导出的可选步骤。
- 对比口径固定：SFT baseline 与 DPO model 使用相同 prompt、generation 参数、随机种子、后处理和评估链路。
- 可追溯优先：候选、偏好对、训练产物和评估报告必须能追溯到原始 train sample、模型路径和配置。

## 5. 输入与输出

### 5.1 输入

DPO 模块需要读取以下输入：

- `SFT/data/train.jsonl`：仅用于候选生成和偏好训练集构建。
- `SFT/data/val.jsonl`：仅用于 DPO 后评估。
- `SFT/data/test.jsonl`：仅用于 DPO 后评估。
- `extracter/data_dict.md`：字段白名单和可用数据表定义。
- `base_model_path`：基础模型目录。
- `sft_adapter_path`：SFT 阶段训练完成的 adapter。
- SFT prompt 构造和输出解析逻辑。
- Extracter 校验逻辑。
- Backtest 数据和回测入口。
- DPO 训练配置。

### 5.2 输出

`DPO/` 模块需要产出至少以下内容：

- `data/candidates.jsonl`：Phase 1 候选生成记录。
- `data/candidate_summary.json`：候选生成统计摘要。
- `data/validated_candidates.jsonl`：Phase 2 校验与回测记录。
- `data/validation_summary.json`：校验与回测统计摘要。
- `data/dpo_train.jsonl`：Phase 3 DPO 偏好训练集。
- `data/hard_cases.jsonl`：无法构建偏好对的样本摘要。
- `data/preference_summary.json`：偏好对构建统计摘要。
- `configs/dpo_config.yaml`：DPO 全流程配置。
- `trained/`：DPO adapter、tokenizer、训练日志和 manifest。
- `output/eval_report.json`：SFT baseline 与 DPO model 的自动评估对比。
- `output/case_studies.md`：人工抽样复核案例和 hard case 分析。

说明：

- 上述路径是模块内建议产物路径，用于约束后续实现方向。
- 本文档阶段只定义规范，不要求这些文件当前已存在。
- 若后续实现需要拆分更细的输出目录，必须保证 README 和 Phase SPEC 同步更新。

## 6. 数据契约

### 6.1 SFT 输入样本契约

DPO 直接复用 SFT 已构建好的 chat JSONL。单条样本结构如下：

```python
{
  "prompt": [
    {"role": "system", "content": "...系统约束..."},
    {"role": "user", "content": "...任务描述..."}
  ],
  "completion": [
    {"role": "assistant", "content": "...目标 JSON 输出..."}
  ],
  "metadata": {
    "sample_id": "...",
    "report_title": "...",
    "report_date": "...",
    "broker": "...",
    "inspiration": "...",
    "class": "...",
    "version": "..."
  }
}
```

DPO 只使用 `prompt` 作为候选生成输入。原始 `completion` 可用于追溯和分析，但不直接作为 chosen 或 rejected。

### 6.2 Assistant 输出契约

DPO 训练中的 `chosen` 和 `rejected` 必须保持与 SFT assistant 输出完全一致：

```python
{
  "reasoning": "...",
  "factor_formula": "...",
  "factor_python": "def compute_factor(...):\n    ...",
  "required_inputs": ["..."],
  "inavailable_inputs": []
}
```

要求：

- 输出必须是合法 JSON 字符串。
- 不附带前后说明文字。
- 不新增 `sample_id`、`report_title`、`ic`、`score` 等训练目标字段。
- `metadata` 只用于训练样本追踪，不进入 assistant 学习目标。
- `factor_python` 必须保持为单个 `compute_factor` 函数。
- `required_inputs` 必须覆盖 `compute_factor` 参数。

### 6.3 候选生成记录

Phase 1 每个候选输出一条记录：

```python
{
  "sample_id": "...",
  "group_id": "...",
  "candidate_id": "...",
  "candidate_index": 0,
  "group_size": 4,
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "raw_output": "...",
  "parsed_json": {
    "reasoning": "...",
    "factor_formula": "...",
    "factor_python": "...",
    "required_inputs": ["..."],
    "inavailable_inputs": []
  },
  "parse_error": None,
  "generation_error": None,
  "metadata": {
    "report_title": "...",
    "report_date": "...",
    "broker": "...",
    "source_split": "train",
    "base_model_path": "...",
    "sft_adapter_path": "..."
  }
}
```

约束：

- `parsed_json` 允许为 `null`。
- `raw_output`、`parse_error` 和 `generation_error` 必须保留，方便定位失败。
- `source_split` 必须为 `train`。
- 候选生成阶段不做 chosen / rejected 判断。

### 6.4 校验与回测记录

Phase 2 每个候选输出一条记录：

```python
{
  "sample_id": "...",
  "group_id": "...",
  "candidate_id": "...",
  "candidate_index": 0,
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "payload": {
    "reasoning": "...",
    "factor_formula": "...",
    "factor_python": "...",
    "required_inputs": ["..."],
    "inavailable_inputs": []
  },
  "validation": {
    "valid_json": true,
    "required_keys": true,
    "valid_field_types": true,
    "validator_pass": true,
    "python_ast_parse": true,
    "required_inputs_arg_match": true,
    "whitelist_compliance": true,
    "errors": []
  },
  "backtest": {
    "success": true,
    "ic_1": 0.05,
    "abs_ic_1": 0.05,
    "rank_ic_1": 0.04,
    "ir_1": 0.7,
    "error": None
  },
  "failure_type": None,
  "failure_severity": 0,
  "metadata": {
    "source_split": "train",
    "report_title": "..."
  }
}
```

`failure_severity = 0` 表示候选通过硬约束并完成回测。不可回测候选不得伪造 `ic_1`、`rank_ic_1` 或 `ir_1`。

失败类型严重度从高到低：

| 严重度 | failure_type | 含义 |
|---:|---|---|
| 8 | `invalid_json` | 无法解析为合法 JSON |
| 7 | `missing_required_keys` | 缺少必要字段 |
| 6 | `invalid_field_type` | 字段类型错误 |
| 5 | `python_ast_error` | `factor_python` 无法通过 `ast.parse` |
| 4 | `arg_mismatch` | `compute_factor` 参数与 `required_inputs` 不一致 |
| 3 | `whitelist_violation` | 使用数据字典外字段、`paused`、分钟级或日内暗含字段 |
| 2 | `validator_error` | 其他 Extracter 校验失败 |
| 1 | `backtest_error` | 结构和代码校验通过，但无法完成回测 |
| 0 | `none` | 通过硬约束并完成回测 |

### 6.5 DPO 偏好训练集

Phase 3 输出 JSONL，每条样本结构如下：

```python
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": "{\"reasoning\": \"...\", \"factor_formula\": \"...\", \"factor_python\": \"...\", \"required_inputs\": [...], \"inavailable_inputs\": [...]}",
  "rejected": "{\"reasoning\": \"...\", \"factor_formula\": \"...\", \"factor_python\": \"...\", \"required_inputs\": [...], \"inavailable_inputs\": [...]}",
  "metadata": {
    "sample_id": "...",
    "report_title": "...",
    "source_split": "train",
    "candidate_group_size": 4,
    "chosen_candidate_id": "...",
    "rejected_candidate_id": "...",
    "chosen_metric": {"ic_1": 0.05, "abs_ic_1": 0.05},
    "rejected_metric": {"ic_1": 0.01, "abs_ic_1": 0.01},
    "preference_rule": "highest_abs_ic_1_vs_lowest_abs_ic_1"
  }
}
```

约束：

- `prompt` 复用 SFT prompt，不重新定义输入格式。
- `chosen` 和 `rejected` 必须是 assistant content 字符串，不是 dict。
- `chosen` 和 `rejected` 只包含 assistant 输出契约中的 5 个字段。
- 所有 DPO pair 必须来自 `SFT/data/train.jsonl`。
- `metadata` 不作为模型输出目标。

### 6.6 Hard Case 记录

无法构建偏好对的候选组写入 hard case：

```python
{
  "sample_id": "...",
  "group_id": "...",
  "report_title": "...",
  "source_split": "train",
  "candidate_count": 4,
  "backtest_success_count": 0,
  "failure_type_counts": {
    "invalid_json": 1,
    "python_ast_error": 2,
    "backtest_error": 1
  },
  "hard_case_reason": "no_backtest_success_candidate"
}
```

Hard case 不保存每个 SFT 生成内容全文，避免输出文件过大并降低误用风险。

### 6.7 训练 Manifest

DPO 训练输出目录必须包含 manifest：

```python
{
  "base_model_path": "...",
  "sft_adapter_path": "...",
  "dpo_adapter_path": "...",
  "merged": false,
  "reference_model": "base_model_path + sft_adapter_path",
  "dpo_train_file": "DPO/data/dpo_train.jsonl",
  "candidate_group_size": 4,
  "preference_rules": [
    "highest_abs_ic_1_vs_lowest_abs_ic_1",
    "single_backtest_success_vs_hard_failure"
  ],
  "trainer_framework": "trl",
  "trainer": "DPOTrainer",
  "seed": 42,
  "created_at": "..."
}
```

## 7. 技术设计

### 7.1 总流程

DPO 全流程分为四个阶段：

1. Candidate Generation：使用 SFT policy 为 train prompt 生成候选池。
2. Validation and Backtest：复用现有校验链路并接入回测指标。
3. Preference Pair Building：根据校验和 1 日 `|IC|` 构建 DPO 偏好对。
4. Training and Evaluation：训练 DPO adapter 并与 SFT baseline 对比。

阶段之间通过 JSONL 文件交接，保证每个阶段都可以独立重跑、抽样检查和统计失败原因。

### 7.2 Phase 1：Candidate Generation

目标：使用已训练的 SFT policy 为 `SFT/data/train.jsonl` 中的每条 prompt 生成多个候选输出，形成后续校验、回测和偏好对构建的候选池。

模型加载：

```text
candidate_policy = base_model_path + sft_adapter_path
```

默认行为：

- 每个 prompt 生成 `g = 4` 个候选。
- 使用可采样生成，确保同一 prompt 能产生候选差异。
- 不加载 DPO adapter。
- 不 merge SFT adapter。
- 不覆盖 SFT adapter。
- 保存 `base_model_path`、`sft_adapter_path`、生成参数和随机种子。

主要步骤：

1. 读取 `SFT/data/train.jsonl`。
2. 对每条记录读取 `prompt` 和 `metadata`。
3. 加载 `base_model_path + sft_adapter_path`。
4. 对每条 prompt 生成多个候选。
5. 对每个候选执行轻量 JSON 解析与规范化。
6. 写出 `DPO/data/candidates.jsonl`。
7. 输出候选生成 summary。

详细设计见 `DPO/SPECS/phase-1-candidate-generation.md`。

### 7.3 Phase 2：Validation and Backtest

目标：对 Phase 1 候选进行结构化校验、字段约束校验、Python 校验和回测，形成可用于偏好排序的候选评分记录。

校验复用约定：

- 结构化解析与字段规范化：优先复用 `SFT/prompt_builder.py`。
- 结构化评估和参数匹配逻辑：优先复用 `SFT/evaluator.py`。
- 字段白名单、不可用字段、日频约束、`paused` 禁用：复用 `extracter.validation.result_validation`。
- Python 语法校验：使用 `ast.parse`。
- 可回测性和 IC 结果：复用 `backtest` 现有链路。

主要步骤：

1. 读取 `DPO/data/candidates.jsonl`。
2. 对 `raw_output` 或 `parsed_json` 执行结构化规范化。
3. 检查必要字段和字段类型。
4. 使用 Extracter 校验字段白名单、不可用字段、日频约束和 `paused` 禁用。
5. 对 `factor_python` 执行 `ast.parse`。
6. 检查 `compute_factor` 参数是否与 `required_inputs` 完全一致。
7. 对通过硬约束的候选执行回测。
8. 记录 1 日 IC、1 日 Rank IC、IR 等摘要指标。
9. 写出 `DPO/data/validated_candidates.jsonl` 和 summary。

详细设计见 `DPO/SPECS/phase-2-validation-and-backtest.md`。

### 7.4 Phase 3：Preference Pair Building

目标：基于 Phase 2 的候选校验与回测结果，自动构建 DPO 偏好训练样本。

偏好规则：

1. 至少两个候选可回测：
   - 过滤出 `backtest.success = true` 的候选。
   - 按 1 日 `|IC|` 从高到低排序。
   - 1 日 `|IC|` 最高的候选作为 chosen。
   - 1 日 `|IC|` 最低的候选作为 rejected。
   - `preference_rule = "highest_abs_ic_1_vs_lowest_abs_ic_1"`。
2. 只有一个候选可回测：
   - 唯一可回测候选作为 chosen。
   - rejected 从同组不可回测候选中选择 `failure_severity` 最高的候选。
   - 若多个失败候选严重度相同，选择候选序号最小者。
   - `preference_rule = "single_backtest_success_vs_hard_failure"`。
3. 没有候选可回测：
   - 不生成 DPO pair。
   - 写入 hard case。
   - `hard_case_reason = "no_backtest_success_candidate"`。

主要步骤：

1. 按 `group_id` 聚合候选。
2. 检查每组是否来自 `source_split = train`。
3. 统计可回测候选数量。
4. 按规则选择 chosen 和 rejected。
5. 将 chosen / rejected payload 序列化为 assistant JSON 字符串。
6. 写出 `DPO/data/dpo_train.jsonl`。
7. 写出 `DPO/data/hard_cases.jsonl`。
8. 输出偏好构建 summary。

详细设计见 `DPO/SPECS/phase-3-preference-pair-building.md`。

### 7.5 Phase 4：Training and Evaluation

目标：基于 Phase 3 的偏好训练集进行 DPO 训练，并在固定评估集上对比 SFT baseline 与 DPO model。

Policy 初始化：

```text
policy_start = base_model_path + sft_adapter_path
```

Reference model：

```text
reference_model = frozen(base_model_path + sft_adapter_path)
```

训练要求：

- policy 与 reference 的起点一致。
- reference 在 DPO 训练期间不更新。
- DPO 训练结果保存到新的 `dpo_adapter_path`。
- 不覆盖 SFT adapter 原目录。
- 默认不进行 adapter merge。
- 训练完成后写出 adapter、tokenizer、日志和 manifest。

主要步骤：

1. 读取 `DPO/data/dpo_train.jsonl`。
2. 加载 `base_model_path + sft_adapter_path` 作为 policy 初始状态。
3. 加载冻结的 `base_model_path + sft_adapter_path` 作为 reference。
4. 使用 DPOTrainer 或等价训练入口进行训练。
5. 保存新的 DPO adapter、tokenizer、日志和 manifest。
6. 使用 SFT baseline 在 `val/test` 上生成预测并评估。
7. 使用 DPO model 在同一批 `val/test` 上生成预测并评估。
8. 输出 `DPO/output/eval_report.json`。

详细设计见 `DPO/SPECS/phase-4-training-and-evaluation.md`。

## 8. Adapter 与模型对比策略

### 8.1 默认训练策略

DPO 默认采用 adapter-on-adapter 的延续训练语义：

1. 加载同一个 `base_model_path`。
2. 加载已有 `sft_adapter_path` 作为 DPO policy 的初始可训练 adapter。
3. 使用 DPO 偏好数据继续训练。
4. 将训练结果保存为新的 `dpo_adapter_path`，例如 `DPO/trained/qwen3_0_6b_dpo_lora`。

该策略不覆盖 SFT 产物，也不要求生成 merged 全量模型。

### 8.2 Reference Model

DPO reference model 使用冻结的 `base_model_path + sft_adapter_path`：

- reference 与 policy 的训练起点一致。
- reference 在 DPO 训练期间不更新。
- reference 用于约束 DPO 不过度偏离 SFT 已学到的格式和领域能力。

### 8.3 Merge 策略

DPO 训练默认不进行 adapter merge：

- 当前项目使用 LoRA / QLoRA，adapter 方式更轻量。
- 现有本地推理后端应支持 `base_model_path + adapter_path`。
- merge 会生成更大的全量模型，QLoRA 场景还可能带来额外精度和显存成本。
- 不 merge 更利于清晰对比 SFT adapter 与 DPO adapter。

merge 只作为未来部署或导出的可选步骤，不作为 DPO 训练前置条件。若未来执行 merge，必须在 manifest 中记录 `merged: true` 和 merged model 路径。

### 8.4 评估对比口径

评估时固定对比两个模型：

```text
SFT baseline = base_model_path + sft_adapter_path
DPO model    = base_model_path + dpo_adapter_path
```

必须固定：

- 相同 tokenizer。
- 相同 prompt。
- 相同 generation 参数。
- 相同随机种子。
- 相同 `SFT/data/val.jsonl` 和 `SFT/data/test.jsonl`。
- 相同结构化解析、字段校验和回测链路。

默认不比较 base model，也不默认比较 merged model。若后续需要 base model 对照，应作为额外实验记录，不影响 SFT vs DPO 主对比。

## 9. 评估设计

### 9.1 自动评估指标

DPO 评估不依赖独立 Eval 模块，主要复用 `extracter`、`SFT` 和 `backtest`。指标包括：

- JSON 成功率。
- required keys 合规率。
- validator pass rate。
- Python AST parse rate。
- `required_inputs` 与 `compute_factor` 参数匹配率。
- 字段白名单合规率。
- 回测通过率。
- 1 日 `|IC|` 均值、中位数、分位数。
- 1 日 Rank IC、IR 等辅助指标。
- DPO 相对 SFT 的指标变化。

### 9.2 对比报告结构

`DPO/output/eval_report.json` 建议包含：

```python
{
  "models": {
    "sft_baseline": {
      "base_model_path": "...",
      "adapter_path": "..."
    },
    "dpo_model": {
      "base_model_path": "...",
      "adapter_path": "..."
    }
  },
  "eval_data": {
    "val_file": "SFT/data/val.jsonl",
    "test_file": "SFT/data/test.jsonl",
    "num_val": 0,
    "num_test": 0
  },
  "generation_config": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "seed": 42
  },
  "metrics": {
    "val": {
      "sft_baseline": {},
      "dpo_model": {},
      "delta": {}
    },
    "test": {
      "sft_baseline": {},
      "dpo_model": {},
      "delta": {}
    }
  },
  "backtest_metrics_available": true,
  "failures": []
}
```

### 9.3 人工抽样复核

自动指标不能完全判断金融逻辑合理性。DPO 阶段应沉淀 `DPO/output/case_studies.md`，用于人工抽样复核：

- DPO 明显优于 SFT 的样本。
- DPO 明显差于 SFT 的样本。
- DPO 与 SFT 都通过结构校验但金融逻辑可疑的样本。
- hard case 中候选全部失败的样本。
- 1 日 `|IC|` 很高但公式或代码可能过拟合的样本。

人工复核不作为偏好训练集生成的阻塞步骤，但用于决定下一轮数据、规则和训练参数调整。

## 10. 配置与 CLI 设计

### 10.1 配置文件

建议使用 `DPO/configs/dpo_config.yaml` 管理全流程配置：

```yaml
paths:
  base_model_path: SFT/model/base
  sft_adapter_path: SFT/trained/qwen3_0_6b_sft_lora
  dpo_adapter_path: DPO/trained/qwen3_0_6b_dpo_lora
  train_file: SFT/data/train.jsonl
  val_file: SFT/data/val.jsonl
  test_file: SFT/data/test.jsonl
  candidates_file: DPO/data/candidates.jsonl
  validated_candidates_file: DPO/data/validated_candidates.jsonl
  dpo_train_file: DPO/data/dpo_train.jsonl
  hard_cases_file: DPO/data/hard_cases.jsonl
  eval_report_file: DPO/output/eval_report.json

candidate_generation:
  group_size: 4
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 1024
  seed: 42

validation:
  require_backtest: true
  ic_horizon: 1

preference:
  primary_metric: abs_ic_1
  allow_single_success_pair: true

training:
  trainer: DPOTrainer
  beta: 0.1
  learning_rate: 0.000005
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  seed: 42

evaluation:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 1024
  seed: 42
```

具体默认值可在实现阶段根据显存和数据规模调整，但配置项语义应保持稳定。

### 10.2 CLI 调用方法

`DPO/cli.py` 已按阶段提供独立入口。所有命令默认读取 `DPO/configs/dpo_config.yaml`，也可以通过 `--config` 指定其他配置文件。

查看总帮助：

```bash
python -m DPO.cli --help
```

查看单个阶段的可用参数：

```bash
python -m DPO.cli generate-candidates --help
python -m DPO.cli validate-candidates --help
python -m DPO.cli build-preferences --help
python -m DPO.cli train --help
python -m DPO.cli evaluate --help
python -m DPO.cli run-all --help
```

#### 10.2.1 Phase 1：生成候选

读取 `SFT/data/train.jsonl`，使用 `base_model_path + sft_adapter_path` 为每条 train prompt 生成候选，输出：

- `DPO/data/candidates.jsonl`
- `DPO/data/candidate_summary.json`

默认调用：

```bash
python -m DPO.cli generate-candidates --config DPO/configs/dpo_config.yaml
```

常用覆盖参数：

```bash
python -m DPO.cli generate-candidates \
  --config DPO/configs/dpo_config.yaml \
  --group-size 4 \
  --seed 42 \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-new-tokens 1024 \
  --overwrite
```

#### 10.2.2 Phase 2：校验与回测

读取 Phase 1 的 `DPO/data/candidates.jsonl`，执行结构校验、字段校验、Python AST 校验、参数匹配、白名单检查和回测，输出：

- `DPO/data/validated_candidates.jsonl`
- `DPO/data/validation_summary.json`

默认调用：

```bash
python -m DPO.cli validate-candidates --config DPO/configs/dpo_config.yaml
```

常用覆盖参数：

```bash
python -m DPO.cli validate-candidates \
  --config DPO/configs/dpo_config.yaml \
  --require-backtest \
  --ic-horizon 1 \
  --overwrite
```

本地只调结构校验、不跑回测时：

```bash
python -m DPO.cli validate-candidates \
  --config DPO/configs/dpo_config.yaml \
  --no-require-backtest \
  --overwrite
```

#### 10.2.3 Phase 3：构建偏好对

读取 Phase 2 的 `DPO/data/validated_candidates.jsonl`，按 `group_id` 构建 `{prompt, chosen, rejected, metadata}` DPO 训练样本，输出：

- `DPO/data/dpo_train.jsonl`
- `DPO/data/hard_cases.jsonl`
- `DPO/data/preference_summary.json`

默认调用：

```bash
python -m DPO.cli build-preferences --config DPO/configs/dpo_config.yaml
```

常用覆盖参数：

```bash
python -m DPO.cli build-preferences \
  --config DPO/configs/dpo_config.yaml \
  --primary-metric abs_ic_1 \
  --allow-single-success-pair \
  --min-abs-ic-gap 0.0 \
  --overwrite
```

禁用“单个可回测候选 vs hard failure”偏好对：

```bash
python -m DPO.cli build-preferences \
  --config DPO/configs/dpo_config.yaml \
  --no-allow-single-success-pair \
  --overwrite
```

#### 10.2.4 Phase 4：DPO 训练

读取 Phase 3 的 `DPO/data/dpo_train.jsonl`，以 `base_model_path + sft_adapter_path` 初始化 policy 和 reference，训练新的 DPO adapter，输出：

- `DPO/trained/<run_name>/`
- `DPO/trained/<run_name>/manifest.json`

默认调用：

```bash
python -m DPO.cli train --config DPO/configs/dpo_config.yaml
```

常用覆盖参数：

```bash
python -m DPO.cli train \
  --config DPO/configs/dpo_config.yaml \
  --beta 0.1 \
  --learning-rate 0.000005 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --overwrite
```

注意：训练不会覆盖 `sft_adapter_path`，`dpo_adapter_path` 必须是独立目录。

#### 10.2.5 Phase 4：SFT vs DPO 评估

在相同 `SFT/data/val.jsonl` 和 `SFT/data/test.jsonl` 上对比：

- SFT baseline：`base_model_path + sft_adapter_path`
- DPO model：`base_model_path + dpo_adapter_path`

输出：

- `DPO/output/eval_report.json`
- `DPO/output/case_studies.md`

默认调用：

```bash
python -m DPO.cli evaluate --config DPO/configs/dpo_config.yaml
```

常用覆盖参数：

```bash
python -m DPO.cli evaluate \
  --config DPO/configs/dpo_config.yaml \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-new-tokens 1024 \
  --seed 42
```

#### 10.2.6 Phase 4：训练后直接评估

`run-all` 当前用于在 Phase 1-3 已完成后顺序执行 Phase 4 的训练和评估：

```bash
python -m DPO.cli run-all --config DPO/configs/dpo_config.yaml
```

允许覆盖训练输出目录存在检查：

```bash
python -m DPO.cli run-all \
  --config DPO/configs/dpo_config.yaml \
  --overwrite
```

#### 10.2.7 推荐执行顺序

从候选生成到最终评估的完整手动流程：

```bash
python -m DPO.cli generate-candidates --config DPO/configs/dpo_config.yaml
python -m DPO.cli validate-candidates --config DPO/configs/dpo_config.yaml
python -m DPO.cli build-preferences --config DPO/configs/dpo_config.yaml
python -m DPO.cli train --config DPO/configs/dpo_config.yaml
python -m DPO.cli evaluate --config DPO/configs/dpo_config.yaml
```

CLI 约定：

- 每个 stage 都可以独立运行。
- 每个 stage 都读取配置文件并支持必要的命令行覆盖。
- 输出文件已存在时默认报错，除非显式传入 overwrite。
- 任一阶段失败必须写出明确错误信息，不能静默丢弃。
- `run-all` 只串联 Phase 4 的 `train` 和 `evaluate`；Phase 1-3 需先显式执行。

## 11. 分期 SPEC 索引

| Phase | 文件 | 目标 | 状态 |
|---|---|---|---|
| Phase 1 | `SPECS/phase-1-candidate-generation.md` | 使用 SFT policy 为 train prompt 生成候选池 | Draft |
| Phase 2 | `SPECS/phase-2-validation-and-backtest.md` | 复用现有校验链路并接入回测指标 | Draft |
| Phase 3 | `SPECS/phase-3-preference-pair-building.md` | 根据校验和 1 日 `|IC|` 构建 DPO 偏好对 | Draft |
| Phase 4 | `SPECS/phase-4-training-and-evaluation.md` | 训练 DPO adapter 并与 SFT baseline 对比 | Draft |

维护规则：

- README 定义跨阶段总契约。
- Phase SPEC 定义阶段内部细节。
- 若 README 与 Phase SPEC 冲突，以 README 为准，并及时修正 Phase SPEC。

## 12. Project Plan

### 12.1 Phase 1：候选生成

状态：Draft，待实现。

任务：

- 设计并实现 DPO 配置读取。
- 复用 SFT prompt 和模型加载能力。
- 实现 train prompt 多候选生成。
- 实现候选 JSON 解析与候选级记录。
- 输出候选生成 summary。

验收标准：

- 所有候选均能追溯到原始 train sample。
- 每个 `group_id` 默认最多包含 4 个候选。
- 所有记录都标明 `source_split = train`。
- 候选生成阶段不做 chosen / rejected 判断。

### 12.2 Phase 2：校验与回测

状态：Draft，待实现。

任务：

- 复用 SFT 和 Extracter 校验逻辑。
- 统一候选校验结果结构。
- 接入 backtest 链路并记录 1 日 IC。
- 定义失败类型与严重度映射。
- 输出校验与回测 summary。

验收标准：

- 每个 Phase 1 候选都产生一条校验记录。
- 每个失败候选都包含明确失败类型。
- 可回测候选必须记录 `ic_1` 和 `abs_ic_1`。
- 不可回测候选不得伪造 IC 指标。

### 12.3 Phase 3：偏好对构建

状态：Draft，待实现。

任务：

- 按 `group_id` 聚合候选。
- 实现多可回测候选偏好规则。
- 实现单可回测候选偏好规则。
- 实现 hard case 输出。
- 序列化 DPO JSONL 训练样本。
- 输出偏好构建 summary。

验收标准：

- 所有 DPO pair 都来自 `SFT/data/train.jsonl`。
- 每条 DPO 样本都包含 `prompt / chosen / rejected / metadata`。
- `chosen` 与 `rejected` 都是字符串，不是 dict。
- 至少两个可回测候选时，chosen 的 1 日 `|IC|` 不低于 rejected。
- 全部不可回测的组不进入 DPO 训练集。

### 12.4 Phase 4：训练与评估

状态：Draft，待实现。

任务：

- 实现 DPO 训练数据加载。
- 实现 policy 与 reference model 加载。
- 接入 DPOTrainer。
- 保存 DPO adapter、tokenizer、训练日志和 manifest。
- 实现 SFT baseline 与 DPO model 的固定口径评估。
- 输出自动评估报告和人工复核案例。

验收标准：

- SFT adapter 原目录未被覆盖。
- DPO adapter 保存到独立目录。
- manifest 明确 `merged: false`。
- manifest 明确 reference model 是冻结的 `base_model_path + sft_adapter_path`。
- SFT baseline 与 DPO model 使用同一批 `val/test` prompt。
- 对比报告包含结构化指标、回测通过率和 1 日 `|IC|` 分布变化。

## 13. 风险与处理策略

### 13.1 偏好噪声

风险：1 日 `|IC|` 可能受样本期、回测数据和偶然性影响，导致偏好对包含噪声。

处理：

- 优先把结构化硬约束作为前置过滤。
- 在 summary 中记录 chosen 与 rejected 的 `abs_ic_1` 差异。
- 对差异很小的 pair 可在后续实现中配置最小 margin。
- 人工抽样复核高 IC 但逻辑可疑的样本。

### 13.2 过度优化格式或回测指标

风险：DPO 可能让模型更偏向短期回测表现，而牺牲金融逻辑解释和泛化能力。

处理：

- reference model 固定为 SFT adapter，约束模型不要过度偏离 SFT。
- 评估中同时观察结构化指标、字段合规率和人工案例。
- 不以单一 `|IC|` 指标作为最终验收。

### 13.3 数据泄漏

风险：误用 `val/test` prompt 构造偏好训练集会污染评估结果。

处理：

- Phase 1 只读取 `SFT/data/train.jsonl`。
- Phase 3 检查每组 `source_split = train`。
- DPO pair metadata 必须记录 `source_split`。
- 评估报告必须记录 val/test 文件路径和样本数量。

### 13.4 Adapter 覆盖

风险：DPO 训练误写入 SFT adapter 目录，导致基线不可复现。

处理：

- DPO 输出目录必须与 SFT adapter 目录不同。
- 训练前检查 `dpo_adapter_path != sft_adapter_path`。
- manifest 明确记录 SFT adapter 和 DPO adapter 路径。

### 13.5 回测链路不可用

风险：本地数据、依赖或回测脚本不可用，导致 Phase 2 或 Phase 4 不能输出回测指标。

处理：

- Phase 2 中回测失败的候选记录为 `backtest_error`。
- Phase 4 若回测链路不可用，仍可输出结构化评估报告。
- 报告必须显式标记 `backtest_metrics_available = false`，不能输出不完整对比结论。

## 14. 后续维护规则

- DPO 需求变更先更新本文档，再更新对应分期 SPEC，最后进入实现。
- 涉及跨阶段输入输出契约的变更，需要同步检查所有后续分期 SPEC。
- 涉及 adapter、reference model 或模型对比口径的变更，必须同步更新 Phase 4。
- 涉及偏好构建规则的变更，必须同步更新 Phase 3 和训练 manifest。
- 涉及评估指标的变更，必须同步更新 Phase 2、Phase 4 和 `eval_report` 结构。
- 每次实现完成后，应把 Project Plan 中对应阶段状态从 Draft 更新为 Implemented 或 Partially Implemented，并补充实际产物路径。
