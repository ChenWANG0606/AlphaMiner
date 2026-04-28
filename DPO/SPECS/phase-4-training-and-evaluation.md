# Phase 4: Training and Evaluation SPEC

## 1. 目标

本阶段负责基于 Phase 3 的偏好训练集进行 DPO 训练，并在固定评估集上对比 SFT baseline 与 DPO model。

本阶段必须明确 adapter 处理方式：默认不 merge SFT adapter 后再训练，而是在 SFT adapter 的基础上继续训练，并将结果另存为新的 DPO adapter。

## 2. 输入

- 基础模型：`base_model_path`
- SFT adapter：`sft_adapter_path`
- DPO 偏好训练集：Phase 3 输出的 JSONL
- 评估集：`SFT/data/val.jsonl` 和 `SFT/data/test.jsonl`
- tokenizer：与 SFT 训练产物一致
- 训练配置：LoRA / QLoRA、batch size、learning rate、epoch、seed 等

## 3. Adapter 策略

### 3.1 Policy 初始化

DPO policy 初始化为：

```text
policy_start = base_model_path + sft_adapter_path
```

训练时：

- 加载同一个 base model。
- 加载 SFT adapter 作为可训练初始 adapter。
- 不覆盖 SFT adapter 原目录。
- 训练结果保存为新的 DPO adapter，例如 `trained/qwen3_0_6b_dpo_lora`。

### 3.2 Reference Model

DPO reference model 为冻结的：

```text
reference_model = base_model_path + sft_adapter_path
```

约束：

- reference 与 policy 的起点一致。
- reference 在训练期间不更新。
- reference 用于约束 DPO model 不过度偏离 SFT 已学到的格式和领域能力。

### 3.3 Merge 策略

DPO 训练默认不进行 adapter merge：

- 当前项目使用 LoRA / QLoRA，adapter 方式更轻量。
- 现有本地推理后端已经支持 `base_model_path + adapter_path`。
- merge 会生成更大的全量模型，QLoRA 场景还可能带来额外精度和显存成本。
- 不 merge 更利于清晰对比 SFT adapter 与 DPO adapter。

merge 只作为未来部署或导出的可选步骤，不作为 DPO 训练前置条件。若未来执行 merge，必须在 manifest 中记录 `merged: true` 和 merged model 路径。

## 4. 训练输出

DPO 训练产物保存到新的输出目录，至少包含：

- DPO adapter。
- tokenizer。
- 训练日志。
- 运行 manifest。

manifest 至少记录：

```python
{
  "base_model_path": "...",
  "sft_adapter_path": "...",
  "dpo_adapter_path": "...",
  "merged": false,
  "reference_model": "base_model_path + sft_adapter_path",
  "dpo_train_file": "...",
  "candidate_group_size": 4,
  "preference_rules": [
    "highest_abs_ic_1_vs_lowest_abs_ic_1",
    "single_backtest_success_vs_hard_failure"
  ],
  "trainer_framework": "trl",
  "trainer": "DPOTrainer"
}
```

## 5. 模型对比口径

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

## 6. 评估指标

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

## 7. 流程

1. 读取 DPO 偏好训练集。
2. 加载 `base_model_path + sft_adapter_path` 作为 policy 初始状态。
3. 加载冻结的 `base_model_path + sft_adapter_path` 作为 reference。
4. 使用 DPOTrainer 或后续等价训练入口进行训练。
5. 保存新的 DPO adapter、tokenizer、日志和 manifest。
6. 使用 SFT baseline 在 `val/test` 上生成预测并评估。
7. 使用 DPO model 在同一批 `val/test` 上生成预测并评估。
8. 输出对比报告。

## 8. 失败处理

- 缺少 `base_model_path`、`sft_adapter_path` 或 DPO 训练集时，停止训练并报告配置错误。
- DPO 训练过程中不得写入或覆盖 SFT adapter 原目录；若输出目录与 SFT adapter 相同，应停止运行。
- reference model 加载失败时，不允许退化为无 reference 训练。
- SFT baseline 或 DPO model 任一评估失败时，对比报告必须标记失败模型和失败阶段，不能输出不完整对比结论。
- 若回测链路不可用，仍可输出结构化评估报告，但必须显式标记 backtest metrics unavailable。

## 9. 验收标准

- SFT adapter 原目录未被覆盖。
- DPO adapter 保存到独立目录。
- manifest 明确 `merged: false`。
- manifest 明确 reference model 是冻结的 `base_model_path + sft_adapter_path`。
- SFT baseline 与 DPO model 使用同一批 `val/test` prompt。
- 对比报告包含结构化指标、回测通过率和 1 日 `|IC|` 分布变化。
