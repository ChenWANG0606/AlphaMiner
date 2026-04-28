# Phase 3: Preference Pair Building SPEC

## 1. 目标

本阶段负责基于 Phase 2 的候选校验与回测结果，自动构建 DPO 偏好训练样本。

DPO 偏好对采用自动构建为主，人工复核只用于抽查和 hard case 分析，不作为训练数据生成的阻塞步骤。

## 2. 输入

- Phase 2 候选校验与回测记录。
- 原始 train prompt 和 metadata。
- 失败类型与严重度定义。

## 3. 输出

### 3.1 DPO 偏好训练集

输出 JSONL，每条样本格式为：

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
    "chosen_metric": {"ic_1": 0.05},
    "rejected_metric": {"ic_1": 0.01},
    "preference_rule": "highest_abs_ic_1_vs_lowest_abs_ic_1"
  }
}
```

约束：

- `prompt` 复用 SFT prompt，不重新定义输入格式。
- `chosen` 和 `rejected` 必须是 assistant content 字符串。
- `chosen` 和 `rejected` 的 JSON 字段限定为 `reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
- `metadata` 不作为模型输出目标。

### 3.2 Hard Case

无法构建偏好对的原始样本写入 `hard_case.json` 或 DPO 后续约定的 output 目录，只保存：

- 原始样本元信息。
- prompt 定位信息。
- 候选数量。
- 失败类型统计。
- 无法构建偏好对的原因。

hard case 不保存每个 SFT 生成内容全文。

## 4. 偏好对构建规则

### 4.1 至少两个候选可回测

- 过滤出 `backtest.success = true` 的候选。
- 按 1 日 `|IC|` 从高到低排序。
- 1 日 `|IC|` 最高的候选作为 chosen。
- 1 日 `|IC|` 最低的候选作为 rejected。
- `preference_rule = "highest_abs_ic_1_vs_lowest_abs_ic_1"`。

### 4.2 只有一个候选可回测

- 唯一可回测候选作为 chosen。
- rejected 从同组不可回测候选中选择 `failure_severity` 最高的候选。
- 若多个失败候选严重度相同，选择候选序号最小者，保证构建结果稳定。
- `preference_rule = "single_backtest_success_vs_hard_failure"`。

### 4.3 没有候选可回测

- 不生成 DPO pair。
- 写入 hard case。
- `hard_case_reason = "no_backtest_success_candidate"`。

## 5. 流程

1. 按 `group_id` 聚合候选。
2. 检查每组是否来自 `source_split = train`。
3. 统计可回测候选数量。
4. 按规则选择 chosen 和 rejected。
5. 将 chosen / rejected payload 序列化为 assistant JSON 字符串。
6. 写出 DPO JSONL 训练样本。
7. 写出 hard case 文件。
8. 输出 summary，包括总组数、pair 数、hard case 数、规则分布、平均 1 日 `|IC|` 差异。

## 6. 失败处理

- 缺少 `group_id`、`prompt` 或 `sample_id` 的候选组不进入 DPO 训练集，写入 hard case。
- 同组候选混入非 train split 时，整组不进入 DPO 训练集，写入 hard case。
- chosen 或 rejected 无法序列化为 assistant JSON 字符串时，整组不进入 DPO 训练集，写入 hard case。
- 同一组无法找到有效 rejected 时，不构造单边样本，写入 hard case。
- hard case 必须记录失败原因，但不得保存完整候选生成内容。

## 7. 验收标准

- 所有 DPO pair 都来自 `SFT/data/train.jsonl`。
- 每条 DPO 样本都包含 `prompt / chosen / rejected / metadata`。
- `chosen` 与 `rejected` 都是字符串，不是 dict。
- 至少两个可回测候选时，chosen 的 1 日 `|IC|` 不低于 rejected。
- 全部不可回测的组不进入 DPO 训练集。
- hard case 中不保存完整候选生成内容。
