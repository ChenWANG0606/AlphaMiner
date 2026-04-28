# Phase 4: Training and Evaluation SPEC

## 1. 文档定位

本文档是 `DPO/` 模块 Phase 4 的分期 SPEC，细化 `DPO/README.md` 中 Training and Evaluation 阶段的实现要求。若本文档与 `DPO/README.md` 的跨阶段契约冲突，以 `DPO/README.md` 为准，并同步修正本文档。

本阶段负责基于 Phase 3 的偏好训练集进行 DPO 训练，并在固定评估集上对比 SFT baseline 与 DPO model。

## 2. 目标

使用 `DPO/data/dpo_train.jsonl` 继续优化 SFT policy，产出新的 DPO adapter，并在 `SFT/data/val.jsonl` 与 `SFT/data/test.jsonl` 上进行固定口径评估。

阶段完成后应产出：

- `DPO/trained/<run_name>/`：DPO adapter、tokenizer、训练日志和 manifest
- `DPO/output/eval_report.json`
- `DPO/output/case_studies.md`

本阶段必须明确 adapter 处理方式：默认不 merge SFT adapter 后再训练，而是在 SFT adapter 的基础上继续训练，并将结果另存为新的 DPO adapter。

## 3. 输入

### 3.1 数据输入

- DPO 偏好训练集：`DPO/data/dpo_train.jsonl`
- 评估集：`SFT/data/val.jsonl`
- 评估集：`SFT/data/test.jsonl`

DPO 偏好训练集每条样本必须包含：

- `prompt`
- `chosen`
- `rejected`
- `metadata.sample_id`
- `metadata.preference_rule`

### 3.2 模型输入

- 基础模型：`paths.base_model_path`
- SFT adapter：`paths.sft_adapter_path`
- tokenizer：与 SFT 训练产物一致

### 3.3 配置输入

配置来源为 `DPO/configs/dpo_config.yaml`。本阶段至少读取：

```yaml
paths:
  base_model_path: SFT/model/base
  sft_adapter_path: SFT/trained/qwen3_0_6b_sft_lora
  dpo_adapter_path: DPO/trained/qwen3_0_6b_dpo_lora
  dpo_train_file: DPO/data/dpo_train.jsonl
  val_file: SFT/data/val.jsonl
  test_file: SFT/data/test.jsonl
  eval_report_file: DPO/output/eval_report.json
  case_studies_file: DPO/output/case_studies.md

training:
  trainer: DPOTrainer
  beta: 0.1
  learning_rate: 0.000005
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_prompt_length: 1536
  max_length: 3072
  seed: 42
  overwrite: false

evaluation:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 1024
  seed: 42
```

实现时可增加 dtype、device_map、load_in_4bit、gradient_checkpointing、logging_steps、save_steps 等运行配置，但不得改变上述字段语义。

## 4. Adapter 策略

### 4.1 Policy 初始化

DPO policy 初始化为：

```text
policy_start = base_model_path + sft_adapter_path
```

训练时：

- 加载同一个 base model。
- 加载 SFT adapter 作为可训练初始 adapter。
- 不覆盖 SFT adapter 原目录。
- 训练结果保存为新的 DPO adapter，例如 `DPO/trained/qwen3_0_6b_dpo_lora`。

### 4.2 Reference Model

DPO reference model 为冻结的：

```text
reference_model = base_model_path + sft_adapter_path
```

约束：

- reference 与 policy 的起点一致。
- reference 在训练期间不更新。
- reference 用于约束 DPO model 不过度偏离 SFT 已学到的格式和领域能力。
- reference model 加载失败时，不允许退化为无 reference 训练。

### 4.3 Merge 策略

DPO 训练默认不进行 adapter merge：

- 当前项目使用 LoRA / QLoRA，adapter 方式更轻量。
- 现有本地推理后端应支持 `base_model_path + adapter_path`。
- merge 会生成更大的全量模型，QLoRA 场景还可能带来额外精度和显存成本。
- 不 merge 更利于清晰对比 SFT adapter 与 DPO adapter。

merge 只作为未来部署或导出的可选步骤，不作为 DPO 训练前置条件。若未来执行 merge，必须在 manifest 中记录 `merged: true` 和 merged model 路径。

## 5. DPO 训练数据适配

### 5.1 原始样本

`DPO/data/dpo_train.jsonl` 的单条样本：

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
    "preference_rule": "highest_abs_ic_1_vs_lowest_abs_ic_1"
  }
}
```

### 5.2 Trainer 输入

DPOTrainer 所需字段应由原始样本转换得到：

```python
{
  "prompt": "...chat template formatted prompt...",
  "chosen": "...assistant JSON string...",
  "rejected": "...assistant JSON string..."
}
```

要求：

- prompt 格式化必须复用 tokenizer 的 chat template 或 SFT 阶段等价模板。
- `chosen` 和 `rejected` 保持 assistant JSON 字符串，不附加解释文本。
- 训练不对 `metadata` 计算损失。
- 若样本的 `chosen` 或 `rejected` 不是字符串，应在训练前数据检查阶段失败。

## 6. 训练输出

DPO 训练产物保存到新的输出目录，至少包含：

- DPO adapter。
- tokenizer。
- 训练日志。
- 运行 manifest。

manifest 文件建议为 `DPO/trained/<run_name>/manifest.json`，至少记录：

```python
{
  "base_model_path": "SFT/model/base",
  "sft_adapter_path": "SFT/trained/qwen3_0_6b_sft_lora",
  "dpo_adapter_path": "DPO/trained/qwen3_0_6b_dpo_lora",
  "merged": false,
  "reference_model": "base_model_path + sft_adapter_path",
  "dpo_train_file": "DPO/data/dpo_train.jsonl",
  "num_train_samples": 100,
  "candidate_group_size": 4,
  "preference_rules": [
    "highest_abs_ic_1_vs_lowest_abs_ic_1",
    "single_backtest_success_vs_hard_failure"
  ],
  "trainer_framework": "trl",
  "trainer": "DPOTrainer",
  "training_config": {
    "beta": 0.1,
    "learning_rate": 0.000005,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "seed": 42
  },
  "created_at": "..."
}
```

约束：

- `dpo_adapter_path` 不得等于 `sft_adapter_path`。
- 输出目录已存在且 `overwrite = false` 时停止运行。
- manifest 必须明确 `merged: false`。

## 7. 模型对比口径

评估时对比两个模型：

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

不默认比较 base model，也不默认比较 merged model。若后续需要 base model 对照，应作为额外实验记录，不影响 SFT vs DPO 主对比。

## 8. 评估指标

评估不依赖独立 Eval 模块，主要复用 `extracter`、`SFT` 和 `backtest`：

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

如果回测链路不可用，仍可输出结构化评估报告，但必须显式标记 `backtest_metrics_available = false`。

## 9. 评估报告

`DPO/output/eval_report.json` 至少包含：

```python
{
  "models": {
    "sft_baseline": {
      "base_model_path": "SFT/model/base",
      "adapter_path": "SFT/trained/qwen3_0_6b_sft_lora"
    },
    "dpo_model": {
      "base_model_path": "SFT/model/base",
      "adapter_path": "DPO/trained/qwen3_0_6b_dpo_lora"
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

Delta 计算规则：

- 率类指标：`dpo_model - sft_baseline`。
- 错误率类指标：仍使用 `dpo_model - sft_baseline`，由报告消费者判断正负含义。
- IC 分布类指标：使用同名统计量相减。

## 10. Case Studies

`DPO/output/case_studies.md` 用于人工抽样复核，建议包含：

- DPO 明显优于 SFT 的样本。
- DPO 明显差于 SFT 的样本。
- DPO 与 SFT 都通过结构校验但金融逻辑可疑的样本。
- hard case 中候选全部失败的样本。
- 1 日 `|IC|` 很高但公式或代码可能过拟合的样本。

Case studies 应引用 `sample_id`、指标摘要和输出片段，不需要保存全量评估原始结果。

## 11. 流程

1. 读取配置文件。
2. 检查 `base_model_path`、`sft_adapter_path` 和 `DPO/data/dpo_train.jsonl`。
3. 检查 `dpo_adapter_path != sft_adapter_path`。
4. 检查输出目录；若已存在且 `overwrite = false`，停止运行。
5. 读取 DPO 偏好训练集并执行数据格式检查。
6. 加载 `base_model_path + sft_adapter_path` 作为 policy 初始状态。
7. 加载冻结的 `base_model_path + sft_adapter_path` 作为 reference。
8. 使用 DPOTrainer 或等价训练入口进行训练。
9. 保存新的 DPO adapter、tokenizer、日志和 manifest。
10. 使用 SFT baseline 在 `val/test` 上生成预测并评估。
11. 使用 DPO model 在同一批 `val/test` 上生成预测并评估。
12. 输出 `DPO/output/eval_report.json`。
13. 输出 `DPO/output/case_studies.md`。

## 12. CLI 行为

建议入口：

```bash
python -m DPO.cli train --config DPO/configs/dpo_config.yaml
python -m DPO.cli evaluate --config DPO/configs/dpo_config.yaml
```

要求：

- `train` 只负责 DPO 训练和 manifest 输出。
- `evaluate` 负责 SFT baseline 与 DPO model 的固定口径评估。
- `run-all` 可在 Phase 1-3 完成后顺序调用 `train` 和 `evaluate`。
- 支持命令行覆盖训练超参数、评估生成参数和 `overwrite`。
- 训练阶段级失败返回非零退出码。
- 评估阶段若单个样本失败，应记录失败并继续评估其他样本；若某个模型整体加载失败，则评估阶段失败。

## 13. 失败处理

- 缺少 `base_model_path`、`sft_adapter_path` 或 DPO 训练集时，停止训练并报告配置错误。
- `dpo_adapter_path == sft_adapter_path` 时，停止训练。
- DPO 训练过程中不得写入或覆盖 SFT adapter 原目录。
- 输出目录已存在且未开启 overwrite 时，停止训练。
- DPO 训练样本缺少 `prompt`、`chosen` 或 `rejected` 时，停止训练并报告数据错误。
- `chosen` 或 `rejected` 不是字符串时，停止训练并报告数据错误。
- reference model 加载失败时，不允许退化为无 reference 训练。
- SFT baseline 或 DPO model 任一评估加载失败时，对比报告必须标记失败模型和失败阶段，不能输出不完整对比结论。
- 若回测链路不可用，仍可输出结构化评估报告，但必须显式标记 `backtest_metrics_available = false`。

## 14. 实现边界

本阶段不得：

- 使用 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 构建 DPO 训练样本。
- 覆盖 SFT adapter 原目录。
- 默认 merge adapter。
- 在 reference model 加载失败时继续无 reference 训练。
- 使用不同的 prompt 或 generation 参数比较 SFT 与 DPO。
- 只评估 DPO 而不评估 SFT baseline。

## 15. 验收标准

- SFT adapter 原目录未被覆盖。
- DPO adapter 保存到独立目录。
- manifest 明确 `merged: false`。
- manifest 明确 reference model 是冻结的 `base_model_path + sft_adapter_path`。
- DPO 训练样本的 `chosen` 和 `rejected` 均为字符串。
- SFT baseline 与 DPO model 使用同一批 `val/test` prompt。
- SFT baseline 与 DPO model 使用相同 generation 参数和随机种子。
- 对比报告包含结构化指标、回测通过率和 1 日 `|IC|` 分布变化。
- 回测不可用时，对比报告显式标记 `backtest_metrics_available = false`。

## 16. 实现计划前测试点

后续实现计划至少应覆盖以下测试：

- `dpo_adapter_path == sft_adapter_path` 时训练阶段失败。
- DPO 训练样本缺少 `chosen` 时训练前数据检查失败。
- `chosen` 为 dict 而不是字符串时训练前数据检查失败。
- manifest 写出 `base_model_path`、`sft_adapter_path`、`dpo_adapter_path`、`merged` 和 `reference_model`。
- 评估阶段对 SFT baseline 与 DPO model 使用相同样本数。
- 评估报告包含 `val`、`test`、`sft_baseline`、`dpo_model` 和 `delta`。
- 回测链路不可用时仍输出结构化指标，并标记 `backtest_metrics_available = false`。
- reference model 加载失败时训练停止。
