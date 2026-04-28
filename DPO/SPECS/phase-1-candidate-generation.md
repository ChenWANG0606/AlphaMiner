# Phase 1: Candidate Generation SPEC

## 1. 目标

本阶段负责使用已训练的 SFT policy 为 `SFT/data/train.jsonl` 中的每条 prompt 生成多个候选输出，形成后续校验、回测和偏好对构建的候选池。

本阶段只负责候选生成和生成级记录，不做偏好排序，不使用 DPO adapter。

## 2. 输入

- 训练样本：`SFT/data/train.jsonl`
- 基础模型：`base_model_path`
- SFT adapter：`sft_adapter_path`
- tokenizer：与 SFT 训练产物一致
- prompt 模板：复用 SFT 的 system / user messages
- 生成配置：
  - 默认候选组大小：`g = 4`
  - 使用可采样生成，确保同一 prompt 能产生候选差异
  - 具体 temperature、top_p、max_new_tokens、seed 后续实现为配置项

## 3. 模型加载约定

候选生成使用当前 SFT policy：

```text
candidate_policy = base_model_path + sft_adapter_path
```

约束：

- 不加载 DPO adapter。
- 不 merge SFT adapter。
- 不覆盖 SFT adapter。
- 生成记录中必须保存 `base_model_path` 和 `sft_adapter_path`，便于复现实验。

## 4. 输出

输出候选生成记录，建议为 JSONL。每条记录对应一个候选，至少包含：

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

`parsed_json` 允许为 `null`，但 `raw_output` 和 `parse_error` 必须保留。

## 5. 流程

1. 读取 `SFT/data/train.jsonl`。
2. 对每条记录读取 `prompt` 和 `metadata`。
3. 加载 `base_model_path + sft_adapter_path`。
4. 对每条 prompt 生成 `g=4` 个候选。
5. 对每个候选执行轻量 JSON 解析与规范化，优先复用 `SFT/prompt_builder.py` 中的输出解析逻辑。
6. 写出候选记录。
7. 输出候选生成 summary，包括输入样本数、目标候选数、实际候选数、解析成功率和失败类型分布。

## 6. 失败处理

- 单个候选生成失败：保留该候选记录，`raw_output` 为空，写入 `parse_error` 或 `generation_error`。
- 单条 prompt 的部分候选失败：不重跑整组，后续阶段按已有候选处理。
- 单条 prompt 的全部候选失败：仍保留失败摘要，后续 Phase 3 进入 hard case。
- 模型加载失败、训练集文件缺失、配置缺失属于阶段级失败，应停止运行。

## 7. 验收标准

- 所有候选均能追溯到原始 train sample。
- 每个 `group_id` 默认最多包含 4 个候选。
- 所有记录都标明 `source_split = train`。
- 文档和后续实现不得从 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 生成 DPO 训练候选。
- 候选生成阶段不做 chosen/rejected 判断。
