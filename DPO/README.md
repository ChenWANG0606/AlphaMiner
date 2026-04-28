# DPO

## 1. 文档定位

本文档是 `DPO/` 模块的总览与索引。更细的阶段设计维护在 `DPO/SPECS/` 下，每个分期 SPEC 可以单独评审、调整和后续实现。

DPO 模块承接总 `SPEC.md` 中的 DPO 设计，目标是在 SFT 模型已经学会基础输出格式和字段约束后，通过自动构建偏好对，让模型更偏向生成结构可用、代码可解析、可回测且 1 日 `|IC|` 更好的因子。

## 2. 模块目标

DPO 阶段的核心目标包括：

- 提升结构化 JSON 输出成功率、字段合规率、Python 可解析率和回测通过率。
- 在通过硬约束的前提下，提升生成因子的 1 日 `|IC|` 表现。
- 保持与 SFT 完全一致的 assistant 输出契约，不新增输出字段。
- 复用 `extracter`、`SFT` 和 `backtest` 的已有校验能力，不引入独立 Eval 模块作为主要依赖。

## 3. 范围定义

### 3.1 In Scope

- 基于 `SFT/data/train.jsonl` 生成 DPO 候选输出。
- 对候选进行结构化解析、字段校验、Python 校验和回测。
- 自动构建 `{prompt, chosen, rejected, metadata}` 格式的 DPO 偏好数据。
- 基于 SFT adapter 继续进行 DPO 训练，并输出新的 DPO adapter。
- 在同一批 `val/test` prompt 上对比 SFT baseline 与 DPO model。

### 3.2 Out of Scope

- 修改 SFT assistant 输出 JSON schema。
- 使用 `SFT/data/val.jsonl` 或 `SFT/data/test.jsonl` 构建 DPO 训练数据。
- 新建独立 Eval 模块替代现有 `extracter` / `SFT` 校验链路。
- 在 DPO 训练前强制 merge SFT adapter。
- 实盘策略组合、交易成本建模和多因子组合优化。

## 4. 核心原则

- 训练数据防泄漏：DPO 偏好数据只来自 `SFT/data/train.jsonl`，`val/test` 只用于评估。
- 输出契约稳定：DPO chosen / rejected 均为 assistant JSON 字符串，字段仍限定为 `reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs`。
- 自动偏好为主：偏好对默认由校验结果和 1 日 `|IC|` 自动生成，人工复核只用于抽查和 hard case 分析。
- adapter 不覆盖：SFT adapter 作为只读基线保留，DPO 训练输出保存为新的 DPO adapter。
- 默认不 merge：DPO 训练不要求先将 SFT adapter merge 到 base model；merge 只作为后续部署或导出的可选步骤。

## 5. Adapter 与模型对比策略

### 5.1 默认训练策略

DPO 默认采用 adapter-on-adapter 的延续训练语义：

1. 加载同一个 `base_model_path`。
2. 加载已有 `sft_adapter_path` 作为 DPO policy 的初始可训练 adapter。
3. 使用 DPO 偏好数据继续训练。
4. 将训练结果保存为新的 `dpo_adapter_path`，例如 `trained/qwen3_0_6b_dpo_lora`。

该策略不覆盖 SFT 产物，也不要求生成 merged 全量模型。

### 5.2 Reference Model

DPO reference model 使用冻结的 `base_model_path + sft_adapter_path`：

- reference 与 policy 的训练起点一致。
- reference 在 DPO 训练期间不更新。
- reference 用于约束 DPO 不过度偏离 SFT 已学到的格式和领域能力。

### 5.3 评估对比口径

评估时固定使用同一套 tokenizer、prompt、generation 参数、随机种子、`val/test` 数据和后处理校验链路：

- SFT baseline：`base_model_path + sft_adapter_path`
- DPO model：`base_model_path + dpo_adapter_path`

默认不比较 merged 模型；若未来为了部署导出 merged 模型，应作为单独导出步骤，并在 manifest 中记录 `merged: true`。

## 6. 分期 SPEC 索引

| Phase | 文件 | 目标 | 状态 |
|---|---|---|---|
| Phase 1 | `SPECS/phase-1-candidate-generation.md` | 使用 SFT policy 为 train prompt 生成候选池 | Draft |
| Phase 2 | `SPECS/phase-2-validation-and-backtest.md` | 复用现有校验链路并接入回测指标 | Draft |
| Phase 3 | `SPECS/phase-3-preference-pair-building.md` | 根据校验和 1 日 `|IC|` 构建 DPO 偏好对 | Draft |
| Phase 4 | `SPECS/phase-4-training-and-evaluation.md` | 训练 DPO adapter 并与 SFT baseline 对比 | Draft |

## 7. 主要产物

DPO 阶段最终应沉淀：

- 候选生成记录：每个 train prompt 的候选摘要、raw output、解析结果和失败类型。
- 校验与回测记录：每个候选的结构化校验、Python 校验、回测状态和 1 日 IC 摘要。
- DPO 偏好训练集：`{prompt, chosen, rejected, metadata}` 格式 JSONL。
- hard case 文件：无法构建偏好对的原始样本元信息和失败摘要。
- DPO 训练产物：DPO adapter、tokenizer、训练日志和运行 manifest。
- DPO 评估报告：SFT baseline 与 DPO model 在 `val/test` 上的指标对比。

## 8. 后续维护规则

- DPO 需求变更先更新本 README 或对应分期 SPEC，再进入实现。
- 涉及跨阶段输入输出契约的变更，需要同步检查所有后续分期 SPEC。
- 涉及 adapter、reference model 或模型对比口径的变更，必须同步更新 Phase 4。
