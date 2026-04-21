# Proj 因子生成Agent

## 项目背景

因子挖掘（Alpha Factor Discovery）是量化投资研究的核心问题之一。传统因子构建主要依赖研究员基于市场经验与经济学直觉，对金融现象进行归纳总结，并进一步形式化为可计算的因子表达式。这一过程通常遵循“**现象观察 → 逻辑推理 → 数学建模 → 回测验证**”的研究范式。尽管该方法在过去数十年中积累了大量有效因子，但其本质上高度依赖专家经验，存在可扩展性差、效率低以及难以系统化的问题。

本项目希望提出一种新的研究框架：**基于推理范式对齐的因子挖掘大模型 Agent**。我们的核心思想是，将量化研究员的思考流程显式建模，并通过大模型进行学习与复现。具体而言，我们的工作包括：

- **（1）基于 SFT 的推理范式注入**：构建“现象—推理及因子公式—Python代码实现”的结构化数据，对模型进行监督微调，使其学习因子构建的思维链；
- **（2）基于 DPO 的偏好对齐**：利用量化研究员对因子优劣的隐式评判标准，构建偏好数据，对模型进行对齐，使其输出符合实际研究偏好；
- **（3）Agent 化因子研究系统**：设计一个能够接受现象与观察，进行推理、因子构建与验证的智能体，实现从研究到落地的自动化流程。

与现有方法相比，本文的关键创新在于：将因子挖掘问题从“搜索问题”转化为“**推理与知识建模问题**”。通过引入结构化推理与人类偏好对齐，我们的方法在提升因子生成效率的同时，也显著增强了结果的可解释性与经济意义。



## 项目设计

1. 实验将包含三个阶段
   - extracter模块：从6G研报数据中提取SFT数据
   - SFT模块：进行SFT注入思考范式
   - DPO模块：进行DPO对齐因子研究偏好

### Extracter

1. 目标：从`研报/` 目录中的研报 PDF 中，沉淀出以“单因子”为单位的高质量样本。方案基于Python进行开发，默认采用“PDF 文本抽取（PyPDF2） + LLM 主流程”的路线。readme文档、代码、中间产物存放在extracter/文件夹下

2. Readme的要求：该 README 不是普通说明文档，而是一个**工程级 Spec + 技术设计文档 + 项目管理文档**，必须满足以下要求：

   - 该 README 同时承担三种角色：

     - **Spec（需求规格说明）**

     - **Technical Design（技术实现设计）**

     - **Project Plan（开发排期与进度管理）**

   - 文档必须支持项目的**持续迭代更新**，每次需求变更必须先更新 Spec，再更新 Design 和 Plan。

3. 数据样本定义标准

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

   - 默认约束：
     - `inspiration` 是因子动机，不是全文摘要。
     - `reasoning` 是“研报证据驱动的推理”，可以提取摘要式逻辑，不要求逐字复刻原文，使用的数据尽量来自data_dict.md
        - 必须基于研报内容做摘要式推理，内容要像因子定义说明，不再承担公式字段
        - 内容要像因子定义说明，而不是读后总结
        - `inspiration` 和 `reasoning`必须直接描述金融现象、因子含义、市场规律和量化逻辑，不允许提到“研报”、“报告”，不允许写“研报指出”“报告认为”“文中提到”“作者认为”“该章节说明”等来源转述语句。
        - 不要总结写作过程，不要评价研报，不要出现“根据研报/从研报来看/本文/本报告/这一章节”等元叙述。
     - `factor_formula` 单独记录因子公式
        - 必须提供清晰、可读的数学表达式或定义
        - 表达式变量名优先与 `required_inputs` 和 `data_dict.md` 中字段保持一致
        - 若采用近似替代，需要与 `reasoning` 和 `inavailable_inputs` 保持一致
     - `factor_python` 由 LLM 根据研报内容生成，不要求研报原文已提供代码。
     - `required_inputs` 必须是可执行层面的字段名或数据表需求。
        - 只可以使用data_dict.md数据字典中提供的字段，如果没有请选择其中可以起到替代作用的
     - `inavailable_inputs`: data_dict.md无法找到但是需要的字段
     - 可追溯性当前主要依赖 `report_title`

   - 样本示例：

     ```python
     {
       "sample_id": "report_id_factor_idx",
       "report_title": "string",
       "report_date": "string|null",
       "broker": "string|null",
       "inspiration": "股价接近过去一年高点时，后续上涨空间可能收窄。",
       "reasoning": "该因子衡量当前价格在过去一年价格区间中的相对位置，值越高表示股价越接近历史高位，潜在阻力越强。",
       "factor_formula": "high_position_252 = close / rolling_max(close, 252)",
       "factor_python": "def compute_factor(close):\\n    high_252 = close.rolling(window=252, min_periods=1).max()\\n    return close / high_252",
       "required_inputs": ["close"],
       "inavailable_inputs": []
     }
     ```

   - 生成的Python代码约束

     - 数据频率限制：生成的因子必须仅依赖日度数据，其他的分钟级、tick 级或其他日内频率数据要纳入`inavailable_inputs`但不要硬造字段名或违规代码。

     - 滚动窗口限制：所提供的数据只有一年，凡是表达式中出现滚动均值、滚动求和、滚动标准差等时间窗口算子，默认窗口不得超出限制一年得限制；若研报原始定义明显依赖更长窗口，优先改写为不超过一年的近似版本，否则标记入`inavailable_inputs`

     - 代码格式约束：禁止使用注释和任何 print / logging 输出，函数名为compute_factor

     - 缺失值处理：表达式中的聚合和算术计算必须显式跳过 `NaN`，优先使用 `pandas` 默认 `skipna=True` 语义或 `numpy` 的 `nanmean`、`nanstd`、`nansum` 等安全写法；不得因为单个缺失值导致整段截面或时间窗口无意义失效。

     - 停牌字段限制：不允许使用 `paused` 字段作为输入或过滤条件；停牌股票已通过价格/成交量字段中的 `NaN` 体现，相关逻辑统一依赖缺失值传播与跳过机制处理。

     - 字段白名单限制：`required_inputs`、`factor_formula` 中直接引用的字段，以及 `factor_python` 中允许引用的字段，必须严格来自文档中列出的字段集合。
       - 行情字段只允许使用 `extracter/data_dict.md` 的 `price` 表日度字段。
       - 基本面字段只允许使用 `extracter/data_dict.md` 中与聚宽文档一致的 `valuation`、`balance`、`income`、`indicator` 表字段。
       - 不允许新增文档外字段名，不允许把 `paused` 继续视为可用字段，不允许以分钟数据衍生出的日频中间字段充当输入

        - 输入为向量化数据（numpy / pandas Series）, 参数是 required_inputs 和 inavailable_inputs 中列出的字段英文名, 每一个输入是一个宽表，行是交易日，列是股票代码,数值是因子值

        - 不允许 for 循环逐股票计算

        - 可使用中间变量（如 vwap = money / volume）

        - 因子计算默认基于“前复权行情数据”

        - 最终因子需可被 reshape 为：
            - index = DatetimeIndex（交易日）
            - columns = 股票代码（如 000001.XSHE / 600000.XSHG）
            
        - 参考形式：
          
            - ```python
                 def compute_factor(close, volume, ...):
                     return factor_df
                 ```

   4. 项目主流程

      - 候选文本筛选阶段

        - PDF解析与文本结构解析

        - PDF筛选与打分

      - 信息抽取预处理阶段

        - PDF解析与文本结构解析
        - LLM因子抽取样本生成

        - 校验与后处理

   5. 主要模块

      - CLI：Extracter模块的入口

        - 输入参数
          - stage：dicovery/generate分别执行对应步骤
          - env-file：默认extracter/.env
          - output path: extracter/默认output
          - max-factors-per-report：传给模型的单篇研报参考因子数量，不作为本地删除阈值
          - max_samples_generation: 仅用于第一阶段，表示生成的 candidates 数量
          - max-concurrency：Generation 阶段同时在途的报告任务数量
          - max-qps：每秒钟最大llm调用的数量

      - pipeline.py ：负责流程编排，可以输入参数选择调用流程

        - Stage1: Discovery
          - 执行顺序：
            - parse pdf：扫描研报/下的PDF，建立文档清单
            - rating reports：文档预处理与质量评估
            - candidate discovery :从全部研报中优先筛出 5 到 10 篇文本层较好、结构清晰、主题明确的报告,在单篇研报内部定位可能包含“因子构建 / 逻辑推导 / 公式定义”的高置信章节或片段。
          - output中生成candiates.csv
        - Stage2: Generation
          - 执行顺序
            - create workers: 为 output/candidates.csv 中的候选报告创建并发 worker，并以 `max-concurrency` 控制在途任务数
            - parse pdf: 每个 worker 内独立解析 PDF
            - async LLM and sample generation：每个 worker 调用兼容 OpenAI Chat Completions 的 LLM 接口，生成结构化样本
            - validation and export：每个 worker 完成校验后，主流程按输入顺序归并结果，再输出人工复核视图和失败记录
        - 实现要求：
          - 候选筛选与样本生成解耦，可分别调用
          - 所有失败显式进入失败记录output/failures.csv，而不是静默丢弃

      - Parser文件夹

        - pdf_parser.py: 使用PyPDF2对pdf进行解析，提取出文本信息
        - Data_dcit_parser.py: 用于解析提供data_dict.md文本，其中用md表格的格式存储提取样本的factor_python的可用字段
        - parser_utils: parser模块的通用utils

      - Validation文件夹

        - Report_rating.py: 实现报告级评分PDF筛选与打分、候选研报排序和章节级候选因子发现

          - 报告级评分：报告级评分的目标不是做学术意义上的排序模型，而是判断“哪篇报告值得进入 Phase 1 的高质量生产流程”。当前评分信号如下：

          | 信号                   | 含义                         | 作用               |
          | ---------------------- | ---------------------------- | ------------------ |
          | `text_extractable`     | 文本是否可抽取               | 保障流程可执行     |
          | `text_length`          | 文本长度是否足够             | 排除信息量不足文档 |
          | `section_signal_count` | 是否存在明显章节结构         | 提高后续定位成功率 |
          | `keyword_signal_count` | 是否包含因子/模型/选股等信号 | 提升主题相关性     |
          | `broker_priority`      | 华泰/海通/东方等优先         | 倾向结构规范报告   |
          | `garble_ratio`         | 乱码比例                     | 过滤低质量文档     |

          - 章节级候选评分章节级评分用于在一篇报告内部找出最可能承载“单因子定义”的片段。当前使用的信号：

          | 信号       | 含义                                       |
          | ---------- | ------------------------------------------ |
          | 正向关键词 | 因子、模型、构建、定义、公式、回归、打分等 |
          | 负向惩罚   | 市场综述、回测、业绩、宏观、年度策略等     |
          | 文本长度   | 片段太短时通常不足以支持完整因子抽取       |

        - Result_valiation.py: 负责把模型输出转化为可交付样本

          - 校验内容
            - 核心字段非空
            - `factor_formula` 必须存在，且是明确数学表达式或定义
            - `factor_formula` 中如直接引用数据字典字段，应与 `required_inputs` 保持一致
            - `factor_python` 可通过 `ast.parse`
            - 函数参数需被 `required_inputs` 覆盖
            - `required_inputs` 必须存在于 `extracter/data_dict.md` 或表级需求中
            - 禁止 `paused`、分钟级/日内数据依赖和文档外字段
            - 对聚合函数的缺失值处理进行 `NaN` 安全检查
          - 最终产生结果sample.jsonl存储在output中
          - 失败的结果记入output/failures.csv

      - LLM_client.py：实现LLM调用和样本生成

        - 调用兼容 OpenAI 的 Chat Completions 接口，按 DeepSeek 接入。使用异步 worker 调度实现并发，设置全局 QPS 限流器，并允许输入并发度参数
        - 设计要点：
          - 设计提示词
          - 提取的数据要符合前文定义的样本标准，包含所有字段
          - 输出强制为结构化 JSON。
          - `required_inputs` 被约束在 `extracter/data_dict.md` 字段空间内。
          - `factor_python` 必须是单个可解析函数，并面向真实量化场景的向量化宽表输入

      - Utils文件夹（可选）：其他对项目有用的辅助文件，比如测试函数，数据结构等

      - Configs.py：项目参数

      - Output文件夹：装在项目的生成文件，需要包含

        - failures.jsonl: 中间任意缓解失败的样本
          - stage：失败的stage
          - report_title：处理失败样本所属于的报告名称
          - reason：处理失败的原因
            - reason_type:
              - PARSE_ERROR
              - NO_FACTOR_FOUND
              - INVALID_JSON
              - INVALID_CODE
              - LOW_QUALITY
              - OTHER
        - candiates.csv：存储Report_rating生成的候选报告名称
          - 候选报告名称
          - 排序
        - samples.jsonl: 最终生成结果json文件，生成的样本服从数据样本定义标准要求

## SFT

1. 监督微调任务目标：在 Extracter 从研报中沉淀出高质量结构化样本后，将“**金融现象理解 → 研究逻辑推导 → 因子公式表达 → Python实现**”这一研究范式注入基础模型，使模型具备面向量化因子研究任务的结构化生成能力。该阶段不是单纯让模型背诵因子模板，而是让模型学习以下能力：

   - 根据输入的市场现象或研究观察(inspiration) 和任务描述，生成合理的因子研究思路；
   - 将研究思路组织成清晰、规范、可追溯的 reasoning / factor_formula / factor_python 输出；
   - 学习领域内常见的因子构建表达方式与金融逻辑；
   - 学习在受限字段空间下进行近似建模，而不是随意发明不可用字段；
   - 学习“输出可执行代码”和“输出经济含义解释”之间的一致性。

2. 训练任务定义

   - 任务形式：SFT 采用 **instruction tuning / chat-style supervised fine-tuning** 方式进行训练。每个样本由“输入任务描述 + 输出目标答案”构成，目标答案为完整结构化结果。

     - 输入：市场现象 / 研究观察 / 约束条件 / 可用字段集合
     - 输出：符合规范的结构化因子定义结果
     - 训练时的目标不是只预测某一个字段，而是预测完整答案，使模型学会在单次响应中同时完成：
       - 动机提炼
       - 金融逻辑推理
       - 数学表达
       - Python实现
       - 输入字段约束对齐

   - 输入信息组成

     - 每条训练样本的用户侧输入原则上由以下部分组成：

       1. **任务描述**

          - 系统提示：

          - 给出一个现象、观察或研究目标`"inspiration": "string"`
          - 例如：价格接近一年高点可能意味着后续上涨空间收窄

       2. **可用字段约束**

          - 来自 extracter/data_dict.md
          - 明确只能使用哪些日频字段与表级字段
          - 和extracter生成样本时的提示词约束进行对齐

       3. **生成要求**

          - 结构化输出
          - 仅使用日度数据
          - 不允许使用分钟级或文档外字段
          - 代码必须为向量化宽表形式

   - 输出目标

     - 模型输出目标

       - ```python
         {
           "reasoning": "string",
           "factor_formula": "string",
           "factor_python": "string",
           "required_inputs": ["string"],
           "inavailable_inputs": ["string"]
         }
         ```

       - 中 SFT 重点学习以下字段之间的一致性：

         - inspiration 与 reasoning 语义一致
         - reasoning 与 factor_formula 定义一致
         - factor_formula 与 factor_python 可相互对应
         - required_inputs 能覆盖 factor_python 参数
         - inavailable_inputs 能反映真实约束，而不是空泛占位

3. 数据构建

   - 数据清洗

     - 进入 SFT 训练集之前，需要对 Extracter 输出样本进行二次清洗。清洗目标不是判断“能否交付”，而是判断“是否适合当作模型模仿对象”。
     - 清洗规则包括：
       - 删除 inspiration、reasoning 明显空泛、套话化的样本
       - 删除 factor_formula 与 factor_python 不一致的样本
       - 删除代码虽然可解析但经济含义明显错误的样本
       - 删除字段约束不自洽的样本
       - 删除严重重复样本，避免模型只学到表面模板
       - 删除过度依赖原文转述语句的样本
       - 删除因子逻辑与日频输入约束明显冲突的样本

   - 数据分层

     - 建议将 SFT 数据按难度分层组织，而不是简单混合训练。可以划分为三类：

       - 基础模式样本：适合让模型先学会输出格式与字段约束，这类样本公式清晰，字段少，代码相对直接，例如：

         - 价格相对位置类

         - 波动率类

         - 成交量占比类

         - 简单量价关系类

       - 中等复杂度样本：用于训练模型处理组合逻辑与近似替代，例如：

         - 多字段组合因子

         - 滚动统计类因子

         - 需要对不可用字段进行替代表达的样本

       - 高复杂度样本：用于训练模型保持研究解释和实现一致，例如：

         - 带有经济含义转译的复杂表达

         - 研报定义较抽象，需要转化为可执行公式的样本

         - 同时涉及输入约束、替代逻辑与实现细节的样本

       - 训练时可采用“先基础、后混合”的 curriculum 方式，提高稳定性。

   - 训练样本格式

     - 推荐统一使用多轮 chat 格式，以便后续与 DPO 阶段共用数据处理接口。单条样本形式如下：

     - ```python
       {
         "messages": [
           {"role": "system", "content": "...系统约束..."},
           {"role": "user", "content": "...任务描述..."},
           {"role": "assistant", "content": "...目标JSON输出..."}
         ]
       }
       ```

     - System Prompt 设计原则

       - System 负责固化长期不变的任务规则，内容包括：
         - 你是量化研究因子生成助手
         - 输出必须为结构化 JSON
         - 只可使用 data_dict.md 中允许的字段
         - 仅依赖日度数据
         - 不得使用 paused
         - reasoning 必须是因子定义说明，而非来源复述
         - factor_python 必须是单个 compute_factor 函数
         - 输入是宽表向量化数据，不允许逐股票 for 循环
       - System 不应过长，避免挤占上下文；项目级硬约束应尽量稳定。

     - User Prompt 组成建议

       - ```python
         任务：根据给定现象与约束，生成单因子样本。
         现象描述：
         ...
         
         可用字段：
         ...
         
         生成要求：
         1. 输出 reasoning / factor_formula / factor_python / required_inputs / inavailable_inputs
         2. 仅使用日频字段
         3. 代码必须为向量化宽表形式
         4. 若原始逻辑依赖不可用字段，请近似替代或写入 inavailable_inputs
         5. 输出 JSON
         ```

     - Assistant 输出格式

       - 固定为 JSON 字符串，不加额外解释文字，减少部署时的不稳定性。例如：

       - ```python
         {
           "sample_id": "...",
           "report_title": "...",
           "report_date": "...",
           "broker": "...",
           "inspiration": "...",
           "reasoning": "...",
           "factor_formula": "...",
           "factor_python": "def compute_factor(...):\n    ...",
           "required_inputs": ["..."],
           "inavailable_inputs": []
         }
         ```

       - 实验结束，推理出的结果和模型输入重新按照样本格式拼接，通过id和原样本拼接成xlsx文件，方便对比校验

4. 训练实现设计

   - 目标产物

     - 训练集、验证集、测试集
     - 训练配置文件
     - 训练脚本
     - 最终 SFT Adapter 或全量模型权重
     - 评估报告
     - 推理 demo 与案例集

   - 数据处理流程

     - 读取 Extracter 输出的 samples.jsonl
     - 执行高质量过滤与重复去重
     - 生成 chat-format 样本
     - 划分 train / val / test
     - 序列化为训练框架需要的数据格式
     - 记录数据版本号与统计信息

   - 数据切分：按“**报告级别切分**”，而不是随机按样本切分。原因是同一篇报告中提取的多个因子样本在语言风格、研究逻辑和背景上高度相关，随机切分会导致泄漏。

     - 建议划分方式：

       - Train：80%

       - Validation：10%

       - Test：10%

     - 保证

       - 同一 report_title 只出现在一个集合中
       - 不同集合中的券商、年份、主题分布尽量均衡

   - 模型选择:

     - Qwen3-0.6B模型

   - 训练方式

     - 采用 **LoRA / QLoRA 的参数高效微调**：
       - 成本低
       - 迭代快
       - 易于与后续 DPO 共用底座
       - 便于保留基础模型通用能力
     - 训练目标采用标准 next-token prediction，但只对 assistant 输出部分计算损失，即：
       - system 和 user 部分作为条件输入
       - assistant 的 JSON 输出为监督目标

   - 损失关注点

     - 虽然底层仍是语言建模损失，但项目关注的不是单纯 perplexity，而是以下几类“有效学习信号”：
       - 输出格式学习
       - 字段一致性学习
       - 金融逻辑到公式的映射学习
       - 公式到代码实现的映射学习
       - 约束条件遵守能力
     - 因此在训练前的数据质量，比单纯增加样本量更重要。

5. 训练配置

   - 输入长度设计：单条样本通常包含

     - system 规则

     - user 任务与字段约束

     - assistant 全量 JSON

   - 因此需要预估上下文长度，避免过度截断。建议：

     - max_prompt_length 保证能完整保留 system + user

     - max_seq_length 能覆盖 assistant 输出中的代码段

   - 若长度受限，优先保留：

     - system 规则

     - 用户现象描述

     - 可用字段约束

     - assistant 输出完整性

     - 不应截断 assistant JSON 的尾部，否则会破坏训练分布。

   - 训练超参数原则

     - 使用较小学习率，避免破坏基础模型语义能力

     - 保持足够的 warmup

     - 使用 validation loss 与结构化指标共同早停

     - 优先稳定收敛，而不是追求极限训练轮数

     - 同步记录数据版本、模型版本、训练配置版本

   - 训练轮次

     - 建议从少量 epoch 起步，根据验证集效果决定是否继续。该任务具有较强模板性和领域局部性，过多 epoch 容易导致：

     - 输出模板僵化

     - 语言表达单一

     - 对训练样本过拟合

     - 见到新现象时泛化能力下降

6. 模型评估

   - 评估设计：SFT 阶段的评估不能只看 loss，必须分成“格式正确性、实现正确性、研究合理性”三层。

   - 自动评估指标

     - 结构化输出成功率：统计模型输出中

       - 是否为合法 JSON

       - 是否包含全部字段

       - 字段类型是否正确

     - 代码可解析率：对 factor_python 执行 ast.parse，统计可解析比例。

     - 字段白名单合规率：检查

       - required_inputs 是否都在白名单中

       - factor_formula 是否引用非法字段

       - factor_python 是否使用非法字段

       - 是否使用 paused 或分钟级暗含字段

     - 参数覆盖率：检查函数参数是否被 required_inputs 覆盖。

     - 约束遵守率：检查最终输入回测接口后可以完成回测的可用比率

   - 语义一致性评估：自动规则之外，还应评估字段间一致性，这部分用规则 + LLM-as-judge 混合评估。

     - inspiration 与 reasoning 是否讲的是同一类现象

     - reasoning 与 factor_formula 是否一致

     - factor_formula 与 factor_python 是否一致

     - required_inputs 与代码参数是否一致

   - 最终验收标准：SFT 阶段应至少满足：

     - 模型能稳定输出合法结构化结果

     - 大部分样本代码可解析并通过规则校验

     - 模型对字段空间约束有稳定遵循能力

     - 模型能从自然语言现象描述生成较合理的因子定义

     - 相较未微调基座模型，在领域任务上有明显提升

7. 推理接口

   - SFT 模型在部署时，主要承担“研究草案生成器”的角色。建议对外提供统一推理接口：

     - 输入

       - 现象描述

       - 可用字段集合

       - 任务约束

     - 输出
       - 结构化因子定义 JSON

8. 主要模块

   - Dataset Builder：负责从 extracter/output/samples.jsonl 构建训练数据
     - 清洗
     - 去重
     - 数据增强
     - 划分 train/val/test
     - 转 chat 格式
   - Prompt Builder：负责构造统一的 system / user / assistant 模板，保证训练与推理输入分布尽量一致
   - Trainer：负责模型加载、LoRA/QLoRA 配置、训练、验证、checkpoint 保存。
   - Evaluator：负责自动评估与案例抽样，包括
     - JSON 合法性
     - 代码可解析性
     - 字段合规性
     - 语义一致性
     - 人工评审样本导出
   - Inference Demo：提供简单脚本，对给定现象或观察生成因子草案，供后续 Agent 集成调用。

9. 输出产物

   ```
   SFT 模块的输出文件建议包括：
   	•	sft/data/train.jsonl
   	•	sft/data/val.jsonl
   	•	sft/data/test.jsonl
   	•	sft/configs/train_config.yaml
   	•	sft/checkpoints/...
   	•	sft/output/eval_report.json
   	•	sft/output/case_studies.md
   ```

   - 其中 eval_report.json 建议至少包含：

     - 数据量统计

     - 训练/验证 loss

     - JSON 成功率

     - 代码可解析率

     - 字段合规率

10. 排期

    - Phase 1：数据准备

      - 完成 Extracter 输出样本的质量复查

      - 建立 SFT 清洗规则

      - 生成第一版 train/val/test

    - Phase 2：训练打通

      - 完成 chat-format 数据构造

      - 跑通基础 SFT 训练

      - 完成最小可用推理 demo

    - Phase 3：评估与迭代

      - 建立自动评估脚本

      - 人工抽样评估案例

      - 分析失败类型并回流修改数据与 prompt

    - Phase 4：为 DPO 做准备

      - 收集 SFT 模型在验证集上的多个候选输出

      - 标注偏好对

      - 沉淀 DPO 数据构建流程

## DPO

1. DPO任务目标：DPO的任务目标在于1.提升生成质量有效质量， 2. 提升生成因子的IC表现。

2. 任务的总体执行流程：
   - DPO样本构建
     - 加载SFT模型，加载训练集样本，按照group（e.g. g = 4）生成样本
     - 按组进行校验
       - 前文进行的各种因子python函数有效检验
       - IC回测
     - 样本对构建
       - 如果组内样本全部能回测出IC结果，选择IC绝对值最高和绝对值最低的构建正负样本对
       - 如果部分可以回测出结果，选择IC绝对值最高的和python函数校验不通过组成正负样本对
       - 如果完全无法回测出结果，抛弃该样本，并且将丢弃的该样本元信息统一保存的output文件夹中的hard_case.json
         - 注意不用保存每个SFT生成的内容，而是只保留原始样本
   - DPO训练
     - 加载所有训练集生成的SFT样本，对DPO进行训练
   - 复用SFT的评估链路
     - 结构化输出成功率：统计模型输出中

       - 是否为合法 JSON

       - 是否包含全部字段

       - 字段类型是否正确
     - 代码可解析率：对 factor_python 执行 ast.parse，统计可解析比例。

     - 字段白名单合规率：检查

       - required_inputs 是否都在白名单中

       - factor_formula 是否引用非法字段

       - factor_python 是否使用非法字段

       - 是否使用 paused 或分钟级暗含字段

     - 参数覆盖率：检查函数参数是否被 required_inputs 覆盖。

     - 约束遵守率：检查最终输入回测接口后可以完成回测的可用比率

## Eval

1. Eval模块从DPO、SFT、Extracter提取出公共的评测api避免重复造轮子


## Backtest
