# Phase 3: Preference Pair Building SPEC

## 1. 文档定位

本文档是 `DPO/` 模块 Phase 3 的分期 SPEC，细化 `DPO/README.md` 中 Preference Pair Building 阶段的实现要求。若本文档与 `DPO/README.md` 的跨阶段契约冲突，以 `DPO/README.md` 为准，并同步修正本文档。

本阶段只基于 Phase 2 输出构建 DPO 偏好训练样本，不重新生成候选、不重新校验候选、不训练模型。

## 2. 目标

基于 Phase 2 的候选校验与回测结果，自动构建 `{prompt, chosen, rejected, metadata}` 格式的 DPO 偏好训练样本。

DPO 偏好对采用自动构建为主，人工复核只用于抽查和 hard case 分析，不作为训练数据生成的阻塞步骤。

阶段完成后应产出：

- `DPO/data/dpo_train.jsonl`
- `DPO/data/hard_cases.jsonl`
- `DPO/data/preference_summary.json`

## 3. 输入

### 3.1 数据输入

- Phase 2 候选校验与回测记录：`DPO/data/validated_candidates.jsonl`

每条记录应包含：

- `sample_id`
- `group_id`
- `candidate_id`
- `candidate_index`
- `group_size`
- `prompt`
- `payload`
- `validation`
- `backtest`
- `failure_type`
- `failure_severity`
- `metadata.source_split`

### 3.2 配置输入

配置来源为 `DPO/configs/dpo_config.yaml`。本阶段至少读取：

```yaml
paths:
  validated_candidates_file: DPO/data/validated_candidates.jsonl
  dpo_train_file: DPO/data/dpo_train.jsonl
  hard_cases_file: DPO/data/hard_cases.jsonl
  preference_summary_file: DPO/data/preference_summary.json

preference:
  primary_metric: abs_ic_1
  allow_single_success_pair: true
  min_abs_ic_gap: 0.0
  overwrite: false
```

`min_abs_ic_gap` 用于过滤两个可回测候选之间差异过小的 pair。默认 `0.0` 表示不启用 gap 过滤。

## 4. 输出

### 4.1 DPO 偏好训练集

`DPO/data/dpo_train.jsonl` 每行对应一个候选组生成的 DPO pair：

```python
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": "{\"reasoning\":\"...\",\"factor_formula\":\"...\",\"factor_python\":\"...\",\"required_inputs\":[\"close\"],\"inavailable_inputs\":[]}",
  "rejected": "{\"reasoning\":\"...\",\"factor_formula\":\"...\",\"factor_python\":\"...\",\"required_inputs\":[\"volume\"],\"inavailable_inputs\":[]}",
  "metadata": {
    "sample_id": "sample_001",
    "group_id": "sample_001",
    "report_title": "...",
    "source_split": "train",
    "candidate_group_size": 4,
    "chosen_candidate_id": "sample_001_cand_00",
    "rejected_candidate_id": "sample_001_cand_03",
    "chosen_metric": {
      "ic_1": 0.05,
      "abs_ic_1": 0.05,
      "rank_ic_1": 0.04,
      "ir_1": 0.7
    },
    "rejected_metric": {
      "ic_1": 0.01,
      "abs_ic_1": 0.01,
      "rank_ic_1": 0.00,
      "ir_1": 0.1
    },
    "preference_rule": "highest_abs_ic_1_vs_lowest_abs_ic_1",
    "abs_ic_1_gap": 0.04
  }
}
```

约束：

- `prompt` 复用 SFT prompt，不重新定义输入格式。
- `chosen` 和 `rejected` 必须是 assistant content 字符串，不是 dict。
- `chosen` 和 `rejected` 的 JSON 字段限定为 `reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
- `chosen` 和 `rejected` 不得包含 `metadata`、`ic`、`score`、`sample_id`、`candidate_id`。
- 所有 DPO pair 必须来自 `SFT/data/train.jsonl`。
- `metadata` 不作为模型输出目标。

### 4.2 Hard Case

无法构建偏好对的候选组写入 `DPO/data/hard_cases.jsonl`：

```python
{
  "sample_id": "sample_001",
  "group_id": "sample_001",
  "report_title": "...",
  "source_split": "train",
  "candidate_count": 4,
  "backtest_success_count": 0,
  "failure_type_counts": {
    "invalid_json": 1,
    "python_ast_error": 2,
    "backtest_error": 1
  },
  "hard_case_reason": "no_backtest_success_candidate",
  "candidate_ids": [
    "sample_001_cand_00",
    "sample_001_cand_01",
    "sample_001_cand_02",
    "sample_001_cand_03"
  ]
}
```

Hard case 约束：

- 不保存每个 SFT 生成内容全文。
- 可以保存 `candidate_ids`，便于回查原始候选文件。
- 必须记录无法构建 pair 的原因。

### 4.3 Summary

`DPO/data/preference_summary.json` 至少包含：

```python
{
  "stage": "preference_pair_building",
  "input_file": "DPO/data/validated_candidates.jsonl",
  "dpo_train_file": "DPO/data/dpo_train.jsonl",
  "hard_cases_file": "DPO/data/hard_cases.jsonl",
  "num_groups": 100,
  "num_pairs": 70,
  "num_hard_cases": 30,
  "rule_counts": {
    "highest_abs_ic_1_vs_lowest_abs_ic_1": 50,
    "single_backtest_success_vs_hard_failure": 20
  },
  "hard_case_reason_counts": {
    "no_backtest_success_candidate": 25,
    "min_abs_ic_gap_not_met": 5
  },
  "mean_abs_ic_1_gap": 0.03,
  "median_abs_ic_1_gap": 0.02,
  "source_split_counts": {
    "train": 100
  }
}
```

若没有 pair，gap 统计字段应为 `null`。

## 5. 偏好对构建规则

### 5.1 分组

- 按 `group_id` 聚合候选。
- 同一组内所有记录必须具有相同 `sample_id`、`prompt` 和 `metadata.source_split`。
- 正式 DPO 训练只接受 `source_split = train` 的候选组。
- 缺少 `group_id`、`sample_id` 或 `prompt` 的记录所在组不得进入 DPO 训练集。

### 5.2 至少两个候选可回测

适用条件：

- 同组内 `backtest.success = true` 的候选数量不少于 2。

选择规则：

1. 过滤出可回测候选。
2. 按 `backtest.abs_ic_1` 从高到低排序。
3. 若 `abs_ic_1` 相同，按 `candidate_index` 从小到大排序。
4. 排序第一的候选作为 `chosen`。
5. 排序最后的候选作为 `rejected`。
6. 计算 `abs_ic_1_gap = chosen.abs_ic_1 - rejected.abs_ic_1`。
7. 若 `abs_ic_1_gap < min_abs_ic_gap`，不生成 pair，写入 hard case。
8. `preference_rule = "highest_abs_ic_1_vs_lowest_abs_ic_1"`。

### 5.3 只有一个候选可回测

适用条件：

- 同组内 `backtest.success = true` 的候选数量等于 1。
- `preference.allow_single_success_pair = true`。

选择规则：

1. 唯一可回测候选作为 `chosen`。
2. 从同组不可回测候选中选择 `failure_severity` 最高的候选作为 `rejected`。
3. 若多个失败候选严重度相同，选择 `candidate_index` 最小者。
4. `abs_ic_1_gap = None`。
5. `preference_rule = "single_backtest_success_vs_hard_failure"`。

若同组没有可用 rejected，写入 hard case。

### 5.4 没有候选可回测

适用条件：

- 同组内 `backtest.success = true` 的候选数量为 0。

行为：

- 不生成 DPO pair。
- 写入 hard case。
- `hard_case_reason = "no_backtest_success_candidate"`。

### 5.5 单成功 pair 被禁用

如果 `allow_single_success_pair = false`，只有一个候选可回测的组不生成 pair，写入 hard case：

```text
hard_case_reason = "single_success_pair_disabled"
```

## 6. Assistant JSON 序列化规则

`chosen` 和 `rejected` 必须由候选的 `payload` 序列化得到。

序列化步骤：

1. 读取候选 `payload`。
2. 只保留字段：`reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
3. 按固定字段顺序构造 dict。
4. 使用 JSON 序列化为字符串。
5. 保留中文字符，不进行 ASCII 转义。
6. 不添加 Markdown、解释文本或代码块围栏。

字段顺序：

```python
[
  "reasoning",
  "factor_formula",
  "factor_python",
  "required_inputs",
  "inavailable_inputs"
]
```

如果 chosen 或 rejected 无法序列化为 assistant JSON 字符串，整组不进入 DPO 训练集，写入 hard case：

```text
hard_case_reason = "assistant_json_serialization_failed"
```

## 7. 流程

1. 读取配置文件。
2. 检查输出路径；若文件已存在且 `overwrite = false`，停止运行。
3. 读取 `DPO/data/validated_candidates.jsonl`。
4. 按 `group_id` 聚合候选。
5. 对每组检查 `sample_id`、`prompt`、`source_split` 一致性。
6. 统计可回测候选数量。
7. 按偏好规则选择 chosen 和 rejected。
8. 将 chosen / rejected payload 序列化为 assistant JSON 字符串。
9. 写出 `DPO/data/dpo_train.jsonl`。
10. 写出 `DPO/data/hard_cases.jsonl`。
11. 写出 `DPO/data/preference_summary.json`。

## 8. CLI 行为

建议入口：

```bash
python -m DPO.cli build-preferences --config DPO/configs/dpo_config.yaml
```

要求：

- 默认读取配置中的路径。
- 支持命令行覆盖 `primary_metric`、`allow_single_success_pair`、`min_abs_ic_gap`、`overwrite`。
- 阶段级失败返回非零退出码。
- 单组无法构建 pair 不导致阶段失败，但必须写入 hard case 和 summary。

## 9. 失败处理

- 配置文件缺失：阶段级失败，停止运行。
- Phase 2 输入文件缺失：阶段级失败，停止运行。
- 输出文件已存在且未开启 overwrite：阶段级失败，停止运行。
- 缺少 `group_id`、`prompt` 或 `sample_id` 的候选组不进入 DPO 训练集，写入 hard case。
- 同组候选混入非 train split 时，整组不进入 DPO 训练集，写入 hard case。
- 同组 prompt 不一致时，整组不进入 DPO 训练集，写入 hard case。
- chosen 或 rejected 无法序列化为 assistant JSON 字符串时，整组不进入 DPO 训练集，写入 hard case。
- 同一组无法找到有效 rejected 时，不构造单边样本，写入 hard case。
- hard case 必须记录失败原因，但不得保存完整候选生成内容。

## 10. 实现边界

本阶段不得：

- 读取 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 构建偏好训练样本。
- 重新生成候选。
- 重新执行回测。
- 修改 Phase 2 校验结果。
- 把候选 `metadata` 写入 `chosen` 或 `rejected` 字符串。
- 构造只有 chosen 或只有 rejected 的单边样本。

## 11. 验收标准

- 所有 DPO pair 都来自 `SFT/data/train.jsonl`。
- 每条 DPO 样本都包含 `prompt / chosen / rejected / metadata`。
- `chosen` 与 `rejected` 都是字符串，不是 dict。
- `chosen` 与 `rejected` 字符串可解析为合法 JSON。
- `chosen` 与 `rejected` JSON 只包含 assistant 输出契约中的 5 个字段。
- 至少两个可回测候选时，chosen 的 1 日 `|IC|` 不低于 rejected。
- 全部不可回测的组不进入 DPO 训练集。
- hard case 中不保存完整候选生成内容。
- `preference_summary.json` 的 pair 数与 `dpo_train.jsonl` 行数一致。

## 12. 实现计划前测试点

后续实现计划至少应覆盖以下测试：

- 两个可回测候选时选择 `abs_ic_1` 最高为 chosen、最低为 rejected。
- `abs_ic_1` 相同候选按 `candidate_index` 稳定排序。
- 只有一个可回测候选时选择最高 `failure_severity` 的失败候选为 rejected。
- `allow_single_success_pair = false` 时单成功组进入 hard case。
- 全部不可回测组进入 hard case。
- 非 train split 组进入 hard case。
- chosen / rejected 序列化后是字符串且可 JSON 解析。
- `min_abs_ic_gap` 未满足时进入 hard case。
- summary 中 pair 数、hard case 数和规则分布正确。
