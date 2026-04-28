# Phase 2: Validation and Backtest SPEC

## 1. 文档定位

本文档是 `DPO/` 模块 Phase 2 的分期 SPEC，细化 `DPO/README.md` 中 Validation and Backtest 阶段的实现要求。若本文档与 `DPO/README.md` 的跨阶段契约冲突，以 `DPO/README.md` 为准，并同步修正本文档。

本阶段只产出候选校验与回测信号，不构建 DPO 训练样本，不做 chosen / rejected 判断。

## 2. 目标

对 Phase 1 产出的候选进行结构化校验、字段约束校验、Python 校验和回测，形成可用于 Phase 3 偏好排序的候选评分记录。

阶段完成后应产出：

- `DPO/data/validated_candidates.jsonl`
- `DPO/data/validation_summary.json`

## 3. 输入

### 3.1 数据输入

- Phase 1 候选生成记录：`DPO/data/candidates.jsonl`
- 数据字典：`extracter/data_dict.md`
- Backtest 所需行情、基本面和收益数据

每条候选记录应包含：

- `sample_id`
- `group_id`
- `candidate_id`
- `candidate_index`
- `prompt`
- `raw_output`
- `parsed_json`
- `metadata.source_split`

### 3.2 代码依赖

DPO 不引入独立 Eval 模块作为主要依赖。本阶段优先复用现有能力：

- 结构化解析与字段规范化：`SFT/prompt_builder.py`
- 结构化评估和参数匹配逻辑：`SFT/evaluator.py`
- 字段白名单、不可用字段、日频约束、`paused` 禁用：`extracter.validation.result_validation`
- Python 语法校验：`ast.parse`
- 可回测性和 IC 结果：`backtest` 现有链路

若某个复用函数当前接口不完全匹配 DPO 输入，实现计划应优先增加薄适配层，不应复制一套独立校验逻辑。

### 3.3 配置输入

配置来源为 `DPO/configs/dpo_config.yaml`。本阶段至少读取：

```yaml
paths:
  candidates_file: DPO/data/candidates.jsonl
  validated_candidates_file: DPO/data/validated_candidates.jsonl
  validation_summary_file: DPO/data/validation_summary.json
  data_dict_file: extracter/data_dict.md

validation:
  require_backtest: true
  ic_horizon: 1
  overwrite: false
```

`require_backtest = false` 只允许用于本地结构化调试。正式 DPO 偏好数据构建应使用可回测记录。

## 4. 输出

### 4.1 校验与回测记录

`DPO/data/validated_candidates.jsonl` 每行对应一个 Phase 1 候选：

```python
{
  "sample_id": "sample_001",
  "group_id": "sample_001",
  "candidate_id": "sample_001_cand_00",
  "candidate_index": 0,
  "group_size": 4,
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "payload": {
    "reasoning": "...",
    "factor_formula": "...",
    "factor_python": "...",
    "required_inputs": ["close"],
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
    "report_title": "...",
    "base_model_path": "...",
    "sft_adapter_path": "..."
  }
}
```

字段约束：

- 每个 Phase 1 候选必须产出一条 Phase 2 记录。
- `payload` 对应 Phase 1 的 `parsed_json`；解析失败时为 `null`。
- 不可回测候选不得伪造 `ic_1`、`abs_ic_1`、`rank_ic_1` 或 `ir_1`。
- `failure_type = None` 且 `failure_severity = 0` 只允许出现在 `backtest.success = true` 的候选。
- `metadata.source_split` 必须沿用 Phase 1，且正式流程必须为 `train`。

### 4.2 Summary

`DPO/data/validation_summary.json` 至少包含：

```python
{
  "stage": "validation_and_backtest",
  "input_file": "DPO/data/candidates.jsonl",
  "output_file": "DPO/data/validated_candidates.jsonl",
  "num_candidates": 400,
  "valid_json_count": 350,
  "required_keys_count": 330,
  "python_ast_parse_count": 310,
  "validator_pass_count": 280,
  "backtest_success_count": 180,
  "backtest_success_rate": 0.45,
  "failure_type_counts": {
    "invalid_json": 50,
    "missing_required_keys": 20,
    "python_ast_error": 20,
    "whitelist_violation": 30,
    "backtest_error": 100
  },
  "ic_1": {
    "mean_abs": 0.03,
    "median_abs": 0.02,
    "p25_abs": 0.01,
    "p75_abs": 0.04
  },
  "rank_ic_1": {
    "mean": 0.01,
    "median": 0.01
  }
}
```

若无任何可回测候选，IC 分布字段应为 `null`，不得输出虚假 0 均值。

## 5. 校验顺序

校验必须按以下顺序执行，并在首次硬失败处设置主 `failure_type`：

1. JSON 结构存在性：`parsed_json` 是否为 dict。
2. 必要字段检查：是否包含 `reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
3. 字段类型检查：
   - `reasoning`：非空字符串
   - `factor_formula`：非空字符串
   - `factor_python`：非空字符串
   - `required_inputs`：字符串列表
   - `inavailable_inputs`：字符串列表
4. Python AST 解析：`factor_python` 是否可通过 `ast.parse`。
5. 函数签名检查：是否存在单个 `compute_factor` 函数。
6. 参数匹配检查：`compute_factor` 参数是否与 `required_inputs` 完全一致。
7. 字段白名单和任务约束检查：
   - 禁止数据字典外字段。
   - 禁止 `paused`。
   - 禁止分钟级、tick 级或其他日内字段。
   - 检查 `required_inputs`、`factor_formula`、`factor_python` 的字段一致性。
8. Extracter 其他 validator 检查。
9. 回测。

软错误可追加进 `validation.errors`，但 Phase 3 使用的主排序失败原因只读取 `failure_type` 和 `failure_severity`。

## 6. 失败类型与严重度

失败类型用于 Phase 3 在“只有一个候选可回测”时选择 rejected。严重度数字越高，越适合作为 hard rejected：

| failure_severity | failure_type | 含义 |
|---:|---|---|
| 8 | `invalid_json` | 无法解析为合法 JSON 或 `parsed_json` 不是 dict |
| 7 | `missing_required_keys` | 缺少必要字段 |
| 6 | `invalid_field_type` | 字段类型错误或核心字符串为空 |
| 5 | `python_ast_error` | `factor_python` 无法通过 `ast.parse` |
| 4 | `arg_mismatch` | `compute_factor` 缺失，或参数与 `required_inputs` 不一致 |
| 3 | `whitelist_violation` | 使用数据字典外字段、`paused`、分钟级或日内暗含字段 |
| 2 | `validator_error` | 其他 Extracter 校验失败 |
| 1 | `backtest_error` | 结构和代码校验通过，但无法完成回测 |
| 0 | `none` | 通过硬约束并完成回测 |

同一候选有多个错误时：

- `failure_type` 记录最高严重度的主错误。
- `validation.errors` 保留所有可收集到的错误。
- 若校验在较早步骤已硬失败，可跳过后续不安全步骤。

## 7. 回测约定

只有通过结构、类型、Python、参数和字段白名单校验的候选进入回测。

回测输入：

- `payload.factor_python`
- `payload.required_inputs`
- backtest 数据源
- `validation.ic_horizon`

回测输出：

- `success`
- `ic_1`
- `abs_ic_1`
- `rank_ic_1`
- `ir_1`
- `error`

约束：

- `abs_ic_1 = abs(ic_1)`。
- 若回测失败，`success = false`，指标字段为 `None`，`error` 写入简短原因。
- 回测失败不影响同组其他候选。
- 若 `validation.require_backtest = false`，记录必须显式标记 `backtest.success = false` 和 `error = "backtest_skipped_by_config"`，该模式下的结果不得进入正式 Phase 3 偏好训练集。

## 8. 流程

1. 读取配置文件。
2. 检查输出路径；若文件已存在且 `overwrite = false`，停止运行。
3. 读取 `DPO/data/candidates.jsonl`。
4. 校验候选记录基本字段。
5. 对每条候选执行校验顺序中的步骤。
6. 对通过硬约束的候选执行回测。
7. 写出 `DPO/data/validated_candidates.jsonl`。
8. 根据全部记录聚合并写出 `DPO/data/validation_summary.json`。

## 9. CLI 行为

建议入口：

```bash
python -m DPO.cli validate-candidates --config DPO/configs/dpo_config.yaml
```

要求：

- 默认读取配置中的路径。
- 支持命令行覆盖 `require_backtest`、`ic_horizon`、`overwrite`。
- 阶段级失败返回非零退出码。
- 单候选失败不导致阶段失败，但必须写入候选级记录和 summary。

## 10. 失败处理

- 配置文件缺失：阶段级失败，停止运行。
- 候选文件缺失：阶段级失败，停止运行。
- 输出文件已存在且未开启 overwrite：阶段级失败，停止运行。
- 候选记录缺少 `candidate_id`、`group_id` 或 `sample_id`：该候选写出失败记录，主错误为 `validator_error`。
- JSON 解析失败的候选不进入 Python 校验和回测。
- 必要字段缺失或字段类型错误的候选不进入回测。
- `ast.parse` 失败的候选不进入回测。
- `compute_factor` 参数与 `required_inputs` 不一致的候选不进入回测。
- 字段白名单或 `paused` 约束违规的候选不进入回测。
- 回测失败的候选必须保留前置校验结果，并记录 `backtest_error`。
- 单个候选失败不得中断同组其他候选校验。

## 11. 实现边界

本阶段不得：

- 构建 DPO pair。
- 写出 chosen / rejected。
- 修改候选生成记录。
- 修改 SFT adapter 或 DPO adapter。
- 重新生成候选。
- 在不可回测候选上填充默认 IC。

## 12. 验收标准

- 每个 Phase 1 候选都产生一条校验记录。
- 每个失败候选都包含明确 `failure_type` 和 `failure_severity`。
- 可回测候选必须记录 `ic_1` 和 `abs_ic_1`。
- 不可回测候选不得伪造 IC 指标。
- `validation_summary.json` 的候选计数与 JSONL 行数一致。
- 本阶段只产出候选评分，不构建 chosen / rejected。

## 13. 实现计划前测试点

后续实现计划至少应覆盖以下测试：

- 合法候选通过校验并进入回测。
- `parsed_json = null` 时输出 `invalid_json`。
- 缺少 `factor_python` 时输出 `missing_required_keys`。
- `required_inputs` 不是 list 时输出 `invalid_field_type`。
- `factor_python` 非法 Python 时输出 `python_ast_error`。
- `compute_factor(close)` 但 `required_inputs = ["volume"]` 时输出 `arg_mismatch`。
- 使用 `paused` 字段时输出 `whitelist_violation`。
- 回测异常时输出 `backtest_error` 且不伪造 IC。
- summary 中 `backtest_success_count` 等于成功记录数。
