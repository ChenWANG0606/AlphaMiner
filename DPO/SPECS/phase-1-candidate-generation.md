# Phase 1: Candidate Generation SPEC

## 1. 文档定位

本文档是 `DPO/` 模块 Phase 1 的分期 SPEC，细化 `DPO/README.md` 中 Candidate Generation 阶段的实现要求。若本文档与 `DPO/README.md` 的跨阶段契约冲突，以 `DPO/README.md` 为准，并同步修正本文档。

本阶段只负责候选生成和生成级记录，不做候选校验、不做回测、不构建 chosen / rejected，不使用 DPO adapter。

## 2. 目标

使用已训练的 SFT policy 为 `SFT/data/train.jsonl` 中的每条 prompt 生成多个候选输出，形成后续校验、回测和偏好对构建的候选池。

阶段完成后应产出：

- `DPO/data/candidates.jsonl`
- `DPO/data/candidate_summary.json`

## 3. 输入

### 3.1 数据输入

- 训练样本：`SFT/data/train.jsonl`
- 每条样本必须包含：
  - `prompt`：由 system / user messages 组成
  - `metadata.sample_id`
  - `metadata.report_title`
  - `metadata.report_date`
  - `metadata.broker`

原始 `completion` 可存在，但本阶段不得把 SFT gold completion 当作候选输出。

### 3.2 模型输入

- 基础模型：`paths.base_model_path`
- SFT adapter：`paths.sft_adapter_path`
- tokenizer：与 SFT 训练产物一致
- prompt 模板：直接复用 `SFT/data/train.jsonl` 中已经构造好的 `prompt`

### 3.3 配置输入

配置来源为 `DPO/configs/dpo_config.yaml`。本阶段至少读取：

```yaml
paths:
  base_model_path: SFT/model/base
  sft_adapter_path: SFT/trained/qwen3_0_6b_sft_lora
  train_file: SFT/data/train.jsonl
  candidates_file: DPO/data/candidates.jsonl
  candidate_summary_file: DPO/data/candidate_summary.json

candidate_generation:
  group_size: 4
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 1024
  seed: 42
  overwrite: false
```

实现时可增加 batch size、device、dtype、load_in_4bit 等运行配置，但不得改变上述字段语义。

## 4. 模型加载约定

候选生成使用当前 SFT policy：

```text
candidate_policy = base_model_path + sft_adapter_path
```

约束：

- 不加载 DPO adapter。
- 不 merge SFT adapter。
- 不覆盖 SFT adapter。
- 生成记录中必须保存 `base_model_path` 和 `sft_adapter_path`。
- 如果 `base_model_path` 或 `sft_adapter_path` 不存在，阶段应停止并报告配置错误。

## 5. 输出

### 5.1 候选记录

`DPO/data/candidates.jsonl` 每行对应一个候选：

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
  "raw_output": "...",
  "parsed_json": {
    "reasoning": "...",
    "factor_formula": "...",
    "factor_python": "...",
    "required_inputs": ["close"],
    "inavailable_inputs": []
  },
  "parse_error": None,
  "generation_error": None,
  "metadata": {
    "report_title": "...",
    "report_date": "...",
    "broker": "...",
    "source_split": "train",
    "base_model_path": "SFT/model/base",
    "sft_adapter_path": "SFT/trained/qwen3_0_6b_sft_lora",
    "generation_config": {
      "temperature": 0.8,
      "top_p": 0.9,
      "max_new_tokens": 1024,
      "seed": 42
    }
  }
}
```

字段约束：

- `group_id` 默认等于 `sample_id`。若同一 `sample_id` 不唯一，实现必须在读入阶段报错，不得隐式覆盖。
- `candidate_id` 格式为 `{group_id}_cand_{candidate_index:02d}`。
- `candidate_index` 从 0 开始。
- `group_size` 等于本轮配置中的目标候选数。
- `parsed_json` 允许为 `null`。
- `raw_output`、`parse_error`、`generation_error` 三者必须保留。
- 生成失败时 `raw_output = ""`，`parsed_json = null`，`generation_error` 写入错误类型和简短原因。
- JSON 解析失败时保留 `raw_output`，`parsed_json = null`，`parse_error` 写入错误类型和简短原因。
- `metadata.source_split` 必须固定为 `train`。

### 5.2 Summary

`DPO/data/candidate_summary.json` 至少包含：

```python
{
  "stage": "candidate_generation",
  "input_file": "SFT/data/train.jsonl",
  "output_file": "DPO/data/candidates.jsonl",
  "base_model_path": "SFT/model/base",
  "sft_adapter_path": "SFT/trained/qwen3_0_6b_sft_lora",
  "group_size": 4,
  "num_input_samples": 100,
  "target_candidate_count": 400,
  "actual_candidate_count": 400,
  "generation_success_count": 398,
  "generation_error_count": 2,
  "parse_success_count": 350,
  "parse_error_count": 48,
  "parse_success_rate": 0.875,
  "error_type_counts": {
    "generation_error": 2,
    "invalid_json": 48
  },
  "seed": 42
}
```

## 6. JSON 解析与规范化

本阶段只做轻量解析，目标是保留后续阶段可用的结构，不承担完整校验。

解析规则：

- 优先复用 `SFT/prompt_builder.py` 中的输出解析逻辑。
- 若 SFT 解析逻辑只支持 completion 格式，实现应封装一个 DPO 侧适配函数，但不要复制粘贴大段解析代码。
- 解析成功只表示输出可转为 dict，并包含可读取的 JSON 对象。
- 不在本阶段判定字段白名单、函数参数、Python AST 或回测可用性。

规范化规则：

- 只保留 assistant 输出契约字段：`reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
- 对缺失字段不做补造；缺失字段应保留在 `parsed_json` 中的真实状态，留给 Phase 2 失败分类。
- 不把 `sample_id`、`report_title`、`ic`、`score` 等字段写入 `parsed_json`。

## 7. 流程

1. 读取配置文件。
2. 检查输出路径；若文件已存在且 `overwrite = false`，停止运行。
3. 读取 `SFT/data/train.jsonl`。
4. 校验每条样本包含 `prompt` 和 `metadata.sample_id`。
5. 检查 `sample_id` 唯一性。
6. 加载 `base_model_path + sft_adapter_path` 和 tokenizer。
7. 对每条 prompt 生成 `group_size` 个候选。
8. 对每个候选执行轻量 JSON 解析与规范化。
9. 按输入样本顺序、候选序号顺序写出 `DPO/data/candidates.jsonl`。
10. 写出 `DPO/data/candidate_summary.json`。

## 8. CLI 行为

建议入口：

```bash
python -m DPO.cli generate-candidates --config DPO/configs/dpo_config.yaml
```

要求：

- 默认读取配置中的路径。
- 支持通过命令行覆盖 `group_size`、`seed`、`temperature`、`top_p`、`max_new_tokens`、`overwrite`。
- 阶段级失败返回非零退出码。
- 单候选失败不导致阶段失败，但必须写入候选记录和 summary。

## 9. 失败处理

- 配置文件缺失：阶段级失败，停止运行。
- 训练集文件缺失：阶段级失败，停止运行。
- `sample_id` 缺失或重复：阶段级失败，停止运行。
- `prompt` 缺失或不是 message list：阶段级失败，停止运行。
- 模型或 adapter 加载失败：阶段级失败，停止运行。
- 输出文件已存在且未开启 overwrite：阶段级失败，停止运行。
- 单个候选生成失败：保留候选记录，`raw_output` 为空，写入 `generation_error`。
- 单个候选 JSON 解析失败：保留 `raw_output`，写入 `parse_error`。
- 单条 prompt 的部分候选失败：不重跑整组，后续阶段按已有候选处理。
- 单条 prompt 的全部候选失败：仍保留失败候选记录，后续 Phase 3 进入 hard case。

## 10. 实现边界

本阶段不得：

- 从 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 读取训练候选。
- 使用 SFT gold completion 直接构造 DPO chosen / rejected。
- 调用 backtest。
- 判断 chosen / rejected。
- 修改 SFT adapter 目录。
- 输出 DPO adapter。

## 11. 验收标准

- 所有候选均能追溯到原始 train sample。
- 每个 `group_id` 默认最多包含 4 个候选，除非配置显式修改 `group_size`。
- 所有记录都标明 `source_split = train`。
- `candidate_id` 在输出文件中唯一。
- `candidate_summary.json` 的候选计数与 JSONL 行数一致。
- 文档和后续实现不得从 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 生成 DPO 训练候选。
- 候选生成阶段不做 chosen / rejected 判断。

## 12. 实现计划前测试点

后续实现计划至少应覆盖以下测试：

- 读取 2 条 train 样本、`group_size = 2` 时输出 4 条候选记录。
- `sample_id` 重复时阶段失败。
- 单个候选输出非法 JSON 时仍保留 `raw_output` 和 `parse_error`。
- 单个候选生成异常时仍保留该候选记录和 `generation_error`。
- 输出文件已存在且 `overwrite = false` 时阶段失败。
- summary 中 `actual_candidate_count` 等于 JSONL 行数。
