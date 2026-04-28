# Phase 2: Validation and Backtest SPEC

## 1. 目标

本阶段负责对 Phase 1 产出的候选进行结构化校验、字段约束校验、Python 校验和回测，形成可用于偏好排序的候选评分记录。

本阶段不构建 DPO 训练样本，只为 Phase 3 提供候选质量信号。

## 2. 输入

- Phase 1 候选生成记录。
- 数据字典：`extracter/data_dict.md`
- SFT 输出解析与规范化逻辑。
- Extracter 校验逻辑。
- Backtest 数据和回测入口。

## 3. 校验复用约定

DPO 不引入独立 Eval 模块作为主要依赖。本阶段优先复用现有能力：

- 结构化解析与字段规范化：`SFT/prompt_builder.py`
- 结构化评估和参数匹配逻辑：`SFT/evaluator.py`
- 字段白名单、不可用字段、日频约束、`paused` 禁用：`extracter.validation.result_validation`
- Python 语法校验：`ast.parse`
- 可回测性和 IC 结果：`backtest` 现有链路

## 4. 输出

输出候选校验与回测记录。每条记录对应一个候选，至少包含：

```python
{
  "sample_id": "...",
  "group_id": "...",
  "candidate_id": "...",
  "candidate_index": 0,
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
    "validator_pass": true,
    "python_ast_parse": true,
    "required_inputs_arg_match": true,
    "whitelist_compliance": true,
    "errors": []
  },
  "backtest": {
    "success": true,
    "ic_1": 0.05,
    "rank_ic_1": 0.04,
    "ir_1": 0.7,
    "error": None
  },
  "failure_severity": 0
}
```

## 5. 失败类型与严重度

失败类型用于 Phase 3 在“只有一个候选可回测”时选择 rejected。默认严重度从高到低：

1. `invalid_json`：无法解析为合法 JSON。
2. `missing_required_keys`：缺少必要字段。
3. `invalid_field_type`：字段类型错误。
4. `python_ast_error`：`factor_python` 无法通过 `ast.parse`。
5. `arg_mismatch`：`compute_factor` 参数与 `required_inputs` 不一致。
6. `whitelist_violation`：使用数据字典外字段、`paused`、分钟级或日内暗含字段。
7. `validator_error`：其他 Extracter 校验失败。
8. `backtest_error`：结构和代码校验通过，但无法完成回测。

`failure_severity = 0` 表示候选通过硬约束并完成回测。

## 6. 流程

1. 读取 Phase 1 候选记录。
2. 对 `raw_output` 或 `parsed_json` 执行结构化规范化。
3. 检查必要字段和字段类型。
4. 使用 Extracter 校验字段白名单、不可用字段、日频约束和 `paused` 禁用。
5. 对 `factor_python` 执行 `ast.parse`。
6. 检查 `compute_factor` 参数是否与 `required_inputs` 完全一致。
7. 对通过硬约束的候选执行回测。
8. 记录 1 日 IC、1 日 Rank IC、IR 等摘要指标。
9. 输出候选级记录和阶段 summary。

## 7. 失败处理

- JSON 解析失败的候选不进入 Python 校验和回测，但必须写出候选级失败记录。
- 必要字段缺失或字段类型错误的候选不进入回测。
- `ast.parse` 失败的候选不进入回测。
- `compute_factor` 参数与 `required_inputs` 不一致的候选不进入回测。
- 字段白名单或 `paused` 约束违规的候选不进入回测。
- 回测失败的候选必须保留前置校验结果，并记录 `backtest_error`。
- 单个候选失败不得中断同组其他候选校验。

## 8. 验收标准

- 每个 Phase 1 候选都产生一条校验记录。
- 每个失败候选都包含明确失败类型。
- 可回测候选必须记录 `ic_1`。
- 不可回测候选不得伪造 IC 指标。
- 本阶段只产出候选评分，不构建 chosen/rejected。
