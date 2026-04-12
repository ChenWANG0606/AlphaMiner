# Extracter

## 1. 文档定位

本文档是 `extracter/` 模块的唯一主文档，同时承担以下三种角色：

- Spec：需求规格说明
- Technical Design：技术实现设计
- Project Plan：开发排期与进度管理

本模块后续所有需求变更，必须先更新本文档中的 Spec，再同步更新 Design 和 Plan，最后才进入实现与联调。

## 2. 模块目标

`extracter` 模块负责从 `研报/` 目录中的券商研报 PDF 中，提取以“单因子”为单位的高质量结构化样本，为后续 SFT 与 DPO 阶段提供训练数据基础。

默认技术路线：

- PDF 文本抽取：`PyPDF2`
- 主流程抽取：兼容 OpenAI Chat Completions 的 LLM 接口
- 开发语言：Python

模块内需要同时沉淀：

- README / 设计文档
- 代码
- 中间产物
- 最终样本

## 3. 范围定义

### 3.1 In Scope

- 扫描 `研报/` 下的 PDF 文件并建立清单
- 对 PDF 的文本可用性与质量进行评分
- 在候选研报中定位高置信的因子相关片段
- 调用 LLM 生成结构化单因子样本
- 对生成结果做字段、代码、输入字段和约束校验
- 输出候选列表、最终样本与失败记录

### 3.2 Out of Scope

- SFT 训练实现
- DPO 训练实现
- 因子回测与收益评估
- 分钟级、Tick 级或任何日内数据驱动的因子落地
- 非 `extracter/data_dict.md` 字段空间外的数据接入

## 4. 核心设计原则

- 先筛选、后生成：候选发现与样本生成必须解耦
- 失败显式记录：任何失败都必须进入失败记录，不能静默丢弃
- 文档优先：需求、设计、计划三者统一维护在本文档
- 白名单驱动：`required_inputs`、`factor_formula` 直接引用的字段以及 `factor_python` 只能引用数据字典允许的字段
- 日频优先：因子实现只允许依赖日度数据，可以降频
- 向量化优先：因子代码必须面向宽表输入，禁止逐股票 `for` 循环
- 可复核：最终输出既支持程序消费，也支持人工复核

## 5. 输入与输出

### 5.1 输入

- 研报 PDF：`../研报/`
- 数据字典：`extracter/data_dict.md`
- 环境配置：默认 `extracter/.env`
- CLI 运行参数

### 5.2 输出

输出目录默认位于 `extracter/output/`，至少包括：

- `candidates.csv`：筛选后的候选报告清单，至少包含报告名称、评分、排序等元信息
- `samples.jsonl`：最终通过校验的样本；新生成样本包含 `factor_formula`，历史导出文件可能仍是旧 schema
- `failures.csv`：失败记录

## 6. 数据样本契约

单条样本的目标结构如下：

```python
{
  "sample_id": "report_id_factor_idx",
  "report_title": "string",
  "report_date": "string|null",
  "broker": "string|null",
  "inspiration": "string",
  "reasoning": "string",
  "factor_formula": "string",
  "factor_python": "string",
  "required_inputs": ["string"],
  "inavailable_inputs": ["string"]
}
```

### 6.1 字段语义

- `sample_id`：单样本唯一标识，默认由报告标识与因子序号组成
- `report_title`：可追溯主键之一，当前追溯主要依赖该字段
- `report_date`：报告日期，只允许从文件名与首页文本提取，无法提取时为 `null`
- `broker`：券商名，只允许从文件名与首页文本提取，无法提取时为 `null`
- `inspiration`：因子动机，不是全文摘要
- `reasoning`：围绕金融现象、量化逻辑和因子定义的摘要式推理，只保留解释逻辑，不重复公式
- `factor_formula`：单独记录因子数学表达式，变量名应尽量与 `required_inputs` 和 `data_dict.md` 字段对齐
- `factor_python`：由 LLM 生成的、可解析的单函数 Python 因子实现
- `required_inputs`：执行该因子所需的具体字段列名
- `inavailable_inputs`：数据字典中找不到、但原始因子逻辑需要的字段

### 6.2 文本内容约束

`inspiration` 与 `reasoning` 必须满足：

- 直接描述金融现象、因子含义、市场规律和量化逻辑
- 不允许出现“研报指出”“报告认为”“文中提到”“作者认为”等来源转述
- 不允许出现“根据研报”“本报告”“本文”“这一章节”等元叙述
- `reasoning` 不是读后总结，而应更接近因子定义说明

`factor_formula` 必须满足：

- 使用独立字段表达，不放回 `reasoning`
- 必须是明确、可读的数学表达式或定义
- 表达式变量名优先使用 `required_inputs` 中的英文名
- 若因子采用近似替代写法，需与 `reasoning` 和 `inavailable_inputs` 保持一致

示例：

```python
{
  "inspiration": "股价接近过去一年高点时，后续上涨空间可能收窄。",
  "reasoning": "该因子衡量当前价格在过去一年价格区间中的相对位置，值越高表示股价越接近历史高位，潜在阻力越强。",
  "factor_formula": "high_position_252 = close / rolling_max(close, 252)",
  "factor_python": "def compute_factor(close):\n    high_252 = close.rolling(window=252, min_periods=1).max()\n    return close / high_252",
  "required_inputs": ["close"],
  "inavailable_inputs": []
}
```

### 6.3 Python 代码约束

`factor_python` 必须满足：

- 函数名固定为 `compute_factor`
- 输入是向量化宽表数据，行是交易日，列是股票代码
- 输出可 reshape 为：
  - `index = DatetimeIndex`
  - `columns = 股票代码`
- 不允许逐股票 `for` 循环
- 可以使用中间变量
- 不允许注释
- 不允许 `print` / `logging`
- 默认基于前复权行情数据

### 6.4 数据频率与窗口约束

- 只允许依赖日度数据
- 分钟级、Tick 级、其他日内数据若被原逻辑依赖，必须进入 `inavailable_inputs`
- 若表达式包含滚动均值、滚动求和、滚动标准差等时间窗口，默认窗口不得超过一年数据范围
- 若原定义明显依赖更长窗口，优先改写为一年内近似版本；无法近似时写入 `inavailable_inputs`

### 6.5 缺失值与字段限制

- 聚合和算术计算必须显式满足 `NaN` 安全
- 优先使用 `pandas` 默认 `skipna=True` 语义或 `numpy.nanmean` / `numpy.nanstd` / `numpy.nansum`
- 不允许使用 `paused` 作为输入或过滤条件
- 行情字段只允许使用 `price` 表日度字段
- 基本面字段只允许使用 `valuation`、`balance`、`income`、`indicator` 表字段
- `required_inputs` 不允许写表名，只允许写具体变量列名
- 不允许新增数据字典外字段名

## 7. 流程设计

模块分为两个可独立执行的阶段。

### 7.1 Stage 1: Discovery

目标：从全量研报中发现优先进入生成流程的高质量候选报告。

执行顺序：

1. `parse pdf`
   - 扫描 `研报/` 下的 PDF
   - 提取文本并建立文档清单
2. `rating reports`
   - 对文档进行预处理与质量评估
3. `candidate discovery`
   - 从全部研报中优先筛出 5 到 10 篇文本层较好、结构清晰、主题明确的报告
   - 在单篇研报内部定位可能包含“因子构建 / 逻辑推导 / 公式定义”的高置信章节或片段
   - 候选片段仅作为报告评分和样本生成时的中间过程数据，不写入 `candidates.csv`

阶段输出：

- `output/candidates.csv`

`candidates.csv` 的最小字段集合要求：

- `report_title`：报告名称
- `score`：报告评分
- `rank`：候选排序

### 7.2 Stage 2: Generation

目标：对候选报告进行结构化因子样本生成与校验导出，候选片段只作为生成阶段的中间输入。

执行顺序：

1. `parse pdf`
   - 读取 `output/candidates.csv` 中的候选项
   - 重新解析相关 PDF 或片段
2. `async llm and sample generation`
   - 调用兼容 OpenAI Chat Completions 的 LLM 接口
   - 并发生成结构化样本
3. `validation and export`
   - 对结构、字段、代码、输入合法性进行校验
   - 输出最终样本与失败记录

阶段输出：

- `output/samples.jsonl`
- `output/failures.csv`

### 7.3 Generation 上下文传递模式

Generation 阶段向 LLM 传递报告上下文时，需要支持两种可切换模式：

1. `section`
   - 仅传递章节级候选片段
   - 片段来源于当前片段发现逻辑筛出的高置信段落
   - 优点是 token 成本更低、主题更聚焦
   - 风险是跨段定义、上下文依赖和符号说明可能被截断
2. `full_text`
   - 传递整篇 PDF 正文文本
   - 默认不包含最后一页
   - 优点是语义上下文更完整，适合定义跨段分布的研报
   - 风险是噪声更高、token 成本更高、提示词控制要求更高

推荐方案：

- 默认使用 `section`
- 当候选片段定位质量差，或因子定义明显依赖跨章节上下文时，手动切换到 `full_text`
- 两种模式只影响传给 LLM 的文本范围，不改变候选报告筛选逻辑和结果校验逻辑

当前不建议默认自动回退：

- 即不建议在 `section` 失败后自动切换到 `full_text`
- 该行为会增加流程不透明性，后续如需要可作为显式增强能力补充

## 8. 模块设计

### 8.1 CLI

`extracter` 模块需要提供统一 CLI 入口。

建议参数如下：

- `--stage`
  - 取值：`discovery` / `generate`
- `--env-file`
  - 默认：`extracter/.env`
- `--output-path`
  - 默认：`extracter/output`
- `--max-factors-per-report`
  - 作为提示词中的参考输出样本数传给模型
  - Generation 阶段不会在本地按该参数截断已返回样本
- `--max-samples-generation`
  - 仅在 Discovery 阶段生效，表示候选数量上限
- `--max-qps`
  - 每秒最大 LLM 调用数量
- `--context-mode`
  - 取值：`section` / `full_text`
  - 默认：`section`
  - 控制 Generation 阶段向 LLM 传递章节级候选片段还是整篇正文

### 8.2 pipeline.py

职责：流程编排与阶段切换。

要求：

- 支持按阶段单独执行
- 支持 Discovery 与 Generation 解耦运行
- 所有失败统一进入失败记录
- 不允许在阶段之间隐式吞错

### 8.3 Parser

目录建议：

- `parser/pdf_parser.py`
  - 使用 `PyPDF2` 对 PDF 进行文本解析
- `parser/data_dict_parser.py`
  - 解析 `extracter/data_dict.md`
  - 提供字段白名单查询
- `parser/parser_utils.py`
  - 解析相关公共方法

### 8.4 Validation

目录建议：

- `validation/report_rating.py`
  - 报告级评分
  - 候选研报排序
  - 章节级候选片段发现
  - 候选文件仅输出报告名称，片段信息不落盘到 `candidates.csv`
- `validation/result_validation.py`
  - 模型输出到可交付样本的转换与校验

### 8.5 LLM Client

建议文件：`llm_client.py`

职责：

- 调用兼容 OpenAI Chat Completions 的接口
- 支持异步并发
- 支持 QPS 限流
- 强制结构化 JSON 输出
- 约束 `required_inputs` 在数据字典字段空间内
- 约束 `factor_python` 为单个可解析函数

### 8.6 Configs

建议文件：`configs.py`

职责：

- 统一维护默认路径
- 统一维护阈值参数、QPS、候选数量和评分阈值
- 统一维护模型与提示词相关配置入口
- 统一维护输出文件命名：`candidates.csv`、`samples.jsonl`、`failures.csv`
- 统一维护 Generation 上下文传递模式的默认值

### 8.7 Utils

目录建议：`utils/`

职责：

- 公共数据结构
- 序列化与导出辅助
- 通用校验工具
- 测试辅助函数

## 9. 评分与校验规则

### 9.1 报告级评分

目标不是学术排序，而是判断“哪篇报告值得进入高质量生产流程”。

当前评分信号：

| 信号 | 含义 | 作用 |
|------|------|------|
| `text_extractable` | 文本是否可抽取 | 保障流程可执行 |
| `text_length` | 文本长度是否足够 | 排除信息量不足文档 |
| `section_signal_count` | 是否存在明显章节结构 | 提高后续定位成功率 |
| `keyword_signal_count` | 是否包含因子/模型/选股等信号 | 提升主题相关性 |
| `broker_priority` | 华泰/海通/东方等优先 | 倾向结构规范报告 |
| `garble_ratio` | 乱码比例 | 过滤低质量文档 |

### 9.2 章节级候选评分

用于在单篇报告内部定位最可能承载“单因子定义”的片段。

当前使用的信号：

| 信号 | 含义 |
|------|------|
| 正向关键词 | 因子、模型、构建、定义、公式、回归、打分等 |
| 负向惩罚 | 市场综述、回测、业绩、宏观、年度策略等 |
| 文本长度 | 片段过短通常不足以支持完整因子抽取 |

### 9.3 样本结果校验

最少需要覆盖以下检查：

- 核心字段非空
- `factor_python` 可通过 `ast.parse`
- 函数参数需被 `required_inputs` 覆盖
- `required_inputs` 必须存在于 `extracter/data_dict.md`
- `required_inputs` 只能是具体字段列名，不能是表名
- 禁止 `paused`
- 禁止分钟级或其他日内数据依赖
- 禁止数据字典外字段
- 聚合函数必须满足 `NaN` 安全要求

## 10. LLM 生成要求

LLM 生成阶段应遵循以下原则：

- 输入必须来自候选报告，且传递给模型的上下文范围是可配置的
- 输出必须是结构化 JSON
- 每个样本必须完整包含所有目标字段
- `inspiration` 与 `reasoning` 必须符合非元叙述约束
- `factor_formula` 必须单独输出，不允许继续附着在 `reasoning` 末尾
- `factor_formula` 中的字段名应尽量与 `required_inputs` 和数据字典字段对齐
- `required_inputs` 必须尽量映射到 `data_dict.md`
- `required_inputs` 只能输出字段列名，不能输出表名
- 无法满足的数据依赖必须明确写入 `inavailable_inputs`
- `factor_python` 必须是面向真实量化宽表输入的可执行函数定义

上下文模式要求：

- `section` 模式下，仅传递章节级候选片段
- `full_text` 模式下，传递整篇正文文本，但默认不包含最后一页
- Prompt 中必须显式声明当前上下文模式，避免模型误判上下文完整性
- 两种模式下输出 JSON 结构、结果校验标准和导出格式完全一致

## 11. 目录规划

当前建议目录结构如下：

```text
extracter/
├── README.md
├── .env
├── data_dict.md
├── configs.py
├── cli.py
├── pipeline.py
├── llm_client.py
├── parser/
│   ├── pdf_parser.py
│   ├── data_dict_parser.py
│   └── parser_utils.py
├── validation/
│   ├── report_rating.py
│   └── result_validation.py
├── utils/
└── output/
    ├── candidates.csv
    ├── samples.jsonl
    └── failures.csv
```

## 12. 开发计划

### 12.1 里程碑

| 里程碑 | 目标 | 状态 |
|------|------|------|
| M0 | 明确 Spec / Design / Plan，初始化 README | Completed |
| M1 | 完成目录骨架与配置约束梳理 | Completed |
| M2 | 完成 PDF 解析与数据字典解析 | Completed |
| M3 | 完成报告级评分与候选发现 | Completed |
| M4 | 完成 LLM 样本生成链路 | In Progress |
| M5 | 完成结果校验与导出 | In Progress |
| M6 | 完成最小可运行链路联调 | In Progress |

### 12.2 交付顺序

1. 固化 README 中的需求、命名和输出规范
2. 建立最小目录结构与配置入口
3. 先打通 Discovery
4. 再打通 Generation
5. 最后补齐验证、失败记录与人工复核视图

### 12.3 进度维护规则

- 每次需求变更必须更新本文档
- 每个里程碑开始前补充输入、输出、验收标准
- 每个里程碑结束后更新状态与实际偏差

## 13. 验收标准

`extracter` 模块的最小验收标准如下：

- 可扫描 `研报/` 目录并建立候选输入
- 可输出候选清单
- 可对候选项调用 LLM 生成结构化样本
- 可校验 `factor_formula` / `factor_python` 的结构合法性与输入字段合法性
- 可输出 `samples.jsonl`
- 任意失败均可在失败记录中定位

## 14. 技术实现细节

本章描述当前 `extracter` 模块已经落地的技术实现方案，重点说明运行时如何组织配置、如何完成 Discovery / Generation，以及失败如何被显式记录。

### 14.1 运行入口与配置装载

- CLI 入口为 `extracter/cli.py`
- 统一通过 `--stage` 选择 `discovery` 或 `generate`
- 统一通过 `build_runtime_config` 构建运行时配置对象
- 运行时配置定义在 `extracter/configs.py`，主要包括：
  - 路径配置：`env_file`、`output_dir`、`report_dir`、`data_dict_path`
  - 流程参数：`max_factors_per_report`、`max_samples_generation`、`max_concurrency`、`max_qps`
  - LLM 配置：`base_url`、`api_key`、`model`、`timeout`、`max_retries`
  - Prompt 配置：`generation_system_prompt`、`generation_user_prompt_template`

当前 `.env` 采用轻量级 key-value 解析，不依赖额外配置库。

### 14.2 Pipeline 编排方式

主流程由 `extracter/pipeline.py` 负责，当前实现遵循“按阶段独立运行”的设计：

1. `run_pipeline`
   - 根据 `stage` 分发到 `run_discovery` 或 `run_generation`
2. `run_discovery`
   - 扫描 `研报/` 下 PDF
   - 解析数据字典
   - 调用报告评分逻辑
   - 输出 `candidates.csv` 和 `failures.csv`
3. `run_generation`
   - 读取 `candidates.csv`
   - 对候选报告重新解析 PDF
   - 提取候选片段
   - 调用 LLM 生成样本
   - 校验结果后输出 `samples.jsonl` 和 `failures.csv`

当前设计中 Discovery 和 Generation 完全解耦，Generation 不会隐式触发 Discovery。

补充说明：

- Discovery 与 Generation 的长耗时报告遍历阶段默认显示 `tqdm` 进度条，便于观察批处理进度

### 14.3 数据字典解析实现

数据字典解析位于 `extracter/parser/data_dict_parser.py`。

实现要点：

- 直接读取 `extracter/data_dict.md`
- 按 `### 表名:` 识别表级块
- 按 Markdown 表格抽取字段名与字段说明
- 最终构造：
  - `tables`：表名到字段集合的映射
  - `allowed_factor_fields`：允许出现在 `required_inputs`、`factor_formula` 直接引用字段和 `factor_python` 中的字段白名单

当前解析时会自动排除：

- 标识字段：`code`、`day`、`pubDate`、`statDate`
- 禁用字段：`paused`

### 14.4 PDF 解析实现

PDF 解析位于 `extracter/parser/pdf_parser.py`。

实现要点：

- 默认使用 `PyPDF2.PdfReader`
- 输出统一结构 `ParsedPdf`
  - `path`
  - `page_count`
  - `first_page_text`
  - `full_text`
- 对抽取出的文本执行空白规范化，避免后续评分受换行噪声影响
- 默认跳过最后一页，不将末页纳入文本抽取
- 若 PDF 只有 1 页，则保留该页，避免整篇文档为空

为了降低环境依赖阻塞，当前额外支持本地 vendor 目录注入：

- 若解释器环境中没有安装 `PyPDF2`
- 会尝试从 `extracter/.vendor/` 加载
- 若仍不可用，则显式抛出 `PdfParseError`

这意味着 PDF 解析失败不会静默吞掉，而会进入失败记录。

### 14.5 Discovery 评分实现

Discovery 评分逻辑位于 `extracter/validation/report_rating.py`。

#### 报告级字段抽取

- `report_date`：通过文件名中的 `YYYYMMDD` 正则抽取
- `broker`：仅通过文件名与首页文本中的券商关键词匹配抽取
- 无法抽取时写为 `null`

#### 报告级评分信号

当前实现的评分由以下信号组合：

- `text_extractable`
- `text_length`
- `section_signal_count`
- `keyword_signal_count`
- `candidate_section_count`
- `broker_priority`
- `garble_ratio`

当前评分方式是启发式加权求和，不是训练得到的排序模型。

#### 候选片段发现

候选片段发现逻辑基于段落粒度：

- 先按空行切分段落
- 统计每段的正向关键词和负向关键词
- 根据 `positive * 2 - negative` 计算片段分数
- 过滤过短片段
- 取分数最高的若干段作为候选片段

候选片段仅用于 Generation 阶段输入，不写入 `candidates.csv`。

#### Discovery 输出

当前 `candidates.csv` 除 README 要求的最小字段外，还额外输出：

- `report_path`
- `report_date`
- `broker`
- `text_length`
- `keyword_signal_count`
- `section_signal_count`
- `candidate_section_count`
- `garble_ratio`

这是为了后续 Generation 直接复用，并便于人工排查评分结果。

### 14.6 LLM 调用实现

LLM 调用位于 `extracter/llm_client.py`。

实现要点：

- 使用标准库 `urllib.request` 直接调用兼容 OpenAI Chat Completions 的接口
- 当前请求体包含：
  - `model`
  - `messages`
  - `temperature`
- 若 `base_url` 未显式包含 `/chat/completions`，会自动补齐
- 当前实现保留 `urllib + asyncio.to_thread`，通过线程安全的全局 `max_qps` 节流配合 worker 并发执行

错误处理策略：

- HTTP 错误：转成 `LLMClientError`
- 网络错误：转成 `LLMClientError`
- 响应 JSON 结构不合法：转成 `LLMClientError`
- 模型文本中提取不到 JSON：转成 `LLMClientError`

### 14.7 Prompt 模板实现

Prompt 模板已从 pipeline 中抽离到 `extracter/configs.py`，便于按需求直接编辑。

当前分为两部分：

- `DEFAULT_GENERATION_SYSTEM_PROMPT`
- `DEFAULT_GENERATION_USER_PROMPT_TEMPLATE`

Generation 阶段会在运行时填充以下变量：

- `report_title`
- `report_date`
- `broker`
- `max_factors_per_report`
- `allowed_fields`
- `context_mode`
- `context_payload`

这样可以在不改业务编排代码的前提下，直接调整提示词策略。

### 14.8 Generation 实现

Generation 主流程位于 `run_generation` 与 `_run_generation_async`。

当前执行顺序如下：

1. 读取 `candidates.csv`
2. 为每份候选报告创建单独 worker，并用 `max_concurrency` 控制在途任务数
3. worker 内重新解析 PDF
4. 根据 `context_mode` 组织 Generation 输入上下文
5. 若为 `section` 模式，则重新发现候选片段
6. 若为 `full_text` 模式，则直接使用整篇正文文本
7. 组装 Prompt 并调用 LLM 获取结构化 JSON
8. 在 worker 内逐条执行结果校验
9. 主流程按输入顺序归并各 worker 结果
10. 通过校验的样本进入 `samples.jsonl`
11. 未通过校验的样本进入 `failures.csv`

建议实现方案：

- 在运行时配置中新增 `context_mode`
- 在运行时配置中新增 `max_concurrency`
- `context_mode=section` 时，沿用当前候选片段发现逻辑
- `context_mode=full_text` 时，直接使用 `ParsedPdf.full_text`
- 两种模式共用同一个 LLM Client、同一个校验器、同一个导出逻辑
- 并发单位是单报告 worker，PDF 解析与 LLM 调用都在 worker 内执行
- 最终 `samples.jsonl` / `failures.csv` 按 `candidates.csv` 输入顺序归并，避免并发导致输出顺序漂移
- `max_qps` 负责全局请求节流，`max_concurrency` 负责同时在途的报告任务数
- Generation 阶段不再使用 `max_samples_generation` 截断候选报告或最终样本
- `max_factors_per_report` 只作为 prompt 提示值传给模型，不再作为本地保留上限
- `full_text` 模式下仍遵循“PDF 默认跳过最后一页”的解析规则
- `section` 模式下若找不到高质量片段，继续记 `NO_FACTOR_FOUND`

当前限制策略：

- Generation 阶段会返回所有通过校验的样本，不再做总样本数截断
- `max_factors_per_report` 不再触发本地删除逻辑

### 14.9 样本校验实现

样本校验位于 `extracter/validation/result_validation.py`。

当前已实现的检查包括：

- 必要字段是否齐全
- `inspiration`、`reasoning`、`factor_formula`、`factor_python` 是否为非空字符串
- `required_inputs` / `inavailable_inputs` 是否为字符串列表
- `required_inputs` 是否全部位于数据字典白名单内
- `required_inputs` 是否包含 `paused`
- `factor_formula` 是否包含明确数学表达式或定义
- `factor_formula` 是否包含未出现在 `required_inputs` 中的数据字典字段
- `factor_formula` 是否包含来源转述式表述
- `factor_python` 是否包含 `print` / `logging`
- `factor_python` 是否能通过 `ast.parse`
- 是否存在 `compute_factor`
- 函数参数集合是否与 `required_inputs` 完全一致
- 函数参数是否在函数体内实际被使用
- `inspiration` / `reasoning` 是否包含来源转述式表述

当前校验仍然偏“结构合法性 + 基本约束合法性”，还没有做到对“因子语义是否忠实于原始片段”进行深层语义校验。

### 14.10 导出与失败记录实现

导出逻辑位于 `extracter/utils/io_utils.py`。

当前实现：

- `write_candidates_csv`
  - 输出候选报告清单
- `write_failures_csv`
  - 输出失败记录
- `write_jsonl`
  - 输出最终样本
- `read_candidates_csv`
  - 供 Generation 阶段读取 Discovery 产物

失败记录统一使用 `FailureRecord` 数据结构，字段为：

- `stage`
- `report_title`
- `reason_type`
- `reason`

当前失败类型已经覆盖：

- `PARSE_ERROR`
- `LOW_QUALITY`
- `NO_FACTOR_FOUND`
- `INVALID_JSON`
- `INVALID_CODE`
- `OTHER`

### 14.11 当前实现边界

当前技术实现已经完成：

- 目录骨架
- 配置装载
- 数据字典解析
- PDF 基础解析
- Discovery 候选报告筛选
- Prompt 模板配置化
- LLM 样本生成主链路
- 基础结果校验
- 输出文件导出

当前仍未完全完成或仍需继续增强的部分包括：

- 更强的因子语义一致性校验
- 更稳定的候选片段定位策略
- 更严格的代码合法性与向量化校验
- LLM 并发调度与重试策略细化
- 更适合人工复核的中间视图输出

## 15. Extracter 调用参数说明与示例

`extracter` 的统一入口为：

```bash
python3 -m extracter.cli ...
```

### 15.1 参数说明

| 参数名 | 是否必填 | 默认值 | 说明 |
|------|------|------|------|
| `--stage` | 是 | 无 | 运行阶段，取值为 `discovery` 或 `generate` |
| `--context-mode` | 否 | `section` | Generation 阶段上下文模式，取值为 `section` 或 `full_text` |
| `--env-file` | 否 | `extracter/.env` | LLM 与运行环境配置文件路径 |
| `--output-path` | 否 | `extracter/output` | 输出目录 |
| `--max-factors-per-report` | 否 | `10` | 传给模型的参考单报告样本数，不触发本地截断 |
| `--max-samples-generation` | 否 | `20` | 仅在 Discovery 阶段生效，表示候选报告数量上限 |
| `--max-concurrency` | 否 | `4` | Generation 阶段最多同时处理的报告 worker 数 |
| `--max-qps` | 否 | `30.0` | LLM 调用的全局最大 QPS |

补充说明：

- `--context-mode` 只在 `--stage generate` 时实际生效
- `--max-concurrency` 只在 `--stage generate` 时实际生效
- `--max-samples-generation` 只在 `--stage discovery` 时实际生效
- `--max-qps` 控制全局请求节流，`--max-concurrency` 控制同时在途任务数
- `--stage discovery` 时，流程会输出 `candidates.csv` 和 `failures.csv`
- `--stage generate` 时，流程依赖 `candidates.csv`，并输出 `samples.jsonl` 和 `failures.csv`

### 15.2 常用调用示例

#### 示例 1：运行 Discovery，筛选候选报告

```bash
python3 -m extracter.cli --stage discovery
```

预期输出：

- `extracter/output/candidates.csv`
- `extracter/output/failures.csv`

#### 示例 2：运行 Discovery，并限制只保留前 5 篇候选报告

```bash
python3 -m extracter.cli --stage discovery --max-samples-generation 5
```

适用场景：

- 想先做小批量候选筛选验证
- 想缩短首次调试时间

#### 示例 3：用章节级候选片段进行样本抽取

```bash
python3 -m extracter.cli --stage generate --context-mode section
```

适用场景：

- 希望降低 token 成本
- 希望让模型更聚焦因子定义片段

#### 示例 4：用全文正文进行样本抽取

```bash
python3 -m extracter.cli --stage generate --context-mode full_text
```

适用场景：

- 候选片段切分不稳定
- 因子定义分散在多个段落或多个章节
- 希望保留更完整的上下文

说明：

- `full_text` 模式下，传给 LLM 的正文默认不包含 PDF 最后一页

#### 示例 5：限制最终生成样本数，并控制单报告样本上限

```bash
python3 -m extracter.cli \
  --stage generate \
  --context-mode full_text \
  --max-factors-per-report 2
```

含义：

- 返回所有通过校验的样本
- `--max-factors-per-report 2` 只作为模型输出数量提示，不会触发本地截断

#### 示例 6：使用自定义环境文件与输出目录

```bash
python3 -m extracter.cli \
  --stage generate \
  --context-mode section \
  --env-file extracter/.env \
  --output-path extracter/output
```

适用场景：

- 切换不同模型配置
- 区分不同实验批次的输出目录

### 15.3 输出结果说明

命令执行结束后，CLI 会打印一段 JSON 结果摘要，当前主要字段包括：

- `stage`：本次执行阶段
- `output_dir`：输出目录
- `candidate_count`：候选数量或最终样本数量
- `failure_count`：失败记录数量
- `candidates_path`：候选文件路径
- `failures_path`：失败文件路径
- `notes`：补充说明

示例：

```json
{
  "stage": "generate",
  "output_dir": "extracter/output",
  "candidate_count": 3,
  "failure_count": 1,
  "candidates_path": "extracter/output/candidates.csv",
  "failures_path": "extracter/output/failures.csv",
  "notes": []
}
```

### 15.4 推荐调用顺序

推荐按以下顺序执行：

1. 先运行 `discovery`
2. 检查 `candidates.csv` 是否符合预期
3. 再运行 `generate`
4. 检查 `samples.jsonl` 与 `failures.csv`

推荐命令：

```bash
python3 -m extracter.cli --stage discovery --max-samples-generation 5
python3 -m extracter.cli --stage generate --context-mode section
```
