from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


DEFAULT_ENV_FILE = ".env"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_STAGE = "discovery"
DEFAULT_CONTEXT_MODE = "section"
DEFAULT_MAX_FACTORS_PER_REPORT = 42
DEFAULT_MAX_SAMPLES_GENERATION = 80
DEFAULT_MAX_CONCURRENCY = 20
DEFAULT_MAX_QPS = 80.0
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_GENERATION_SYSTEM_PROMPT = dedent(
"""
你是量化因子抽取助手，目标是从候选文本中构造“可计算单因子”。

你必须严格遵守以下规则：

【输出格式】
- 只能输出 JSON，不要输出 markdown
- 顶层结构必须是：
  {"samples": [...]}

【字段要求】
每个 sample 必须包含：
- inspiration
- reasoning
- factor_formula
- factor_python
- required_inputs
- inavailable_inputs

【inspiration / reasoning / factor_formula】
- 必须直接描述金融现象、市场规律、量化逻辑
- 禁止出现任何来源描述或元叙述：
  禁止：研报、报告、本文、作者认为、文中提到等
- reasoning 必须是“推导式说明”，只负责解释因子逻辑与定义，不要在末尾重复公式
- factor_formula 必须单独给出明确、可读的数学表达式或定义
- factor_formula 中的变量名优先使用 required_inputs 中的英文名，并尽量与数据字典字段一致
- 若因子采用近似替代，factor_formula、reasoning 与 inavailable_inputs 三者必须相互一致

【required_inputs】
- 只能使用给定字段白名单中的字段名（如 close, volume, money 等）
- 必须是“实际计算中用到的字段”
- 严禁出现未使用字段
- 严禁默认只使用 close，必须根据因子逻辑选择字段

【factor_python】
- 必须定义函数：def compute_factor(...)
- 参数必须与 required_inputs 完全一致（顺序也一致）
- 每个 required_inputs 必须在函数中被实际使用
- 输入的每个变量都是一个 DataFrame（index=日期, columns=股票）
- 输出必须是同结构 DataFrame

【计算约束】
- 仅允许日频数据
- 禁止分钟/tick/日内数据
- 禁止使用 paused
- 不允许 for 循环逐股票计算
- 允许向量化计算（pandas / numpy）
- 滚动窗口 ≤ 252日（约一年） 或者 12个月 
- 所有计算必须能处理 NaN（默认 skipna 或 nan* 系列）

【代码风格】
- 禁止 print / logging
- 禁止在代码中写注释
- 可使用中间变量

【inavailable_inputs】
- 若因子必须依赖但白名单中没有 → 填写类别（中文）
- 能替代则不要填

【核心原则】
这是“因子定义生成任务”，不是文本总结任务。
"""
).strip()
DEFAULT_GENERATION_USER_PROMPT_TEMPLATE = dedent(
    """
    报告标题: {report_title}
    报告日期: {report_date}
    券商: {broker}
    上下文模式: {context_mode}
    参考输出样本数: {max_factors_per_report}

    输出格式必须是 JSON 对象，形如：
    {{"samples": [{{"inspiration": "...", "reasoning": "...", "factor_formula": "...", "factor_python": "...", "required_inputs": ["close"], "inavailable_inputs": []}}]}}

    允许使用的字段白名单:
    {allowed_fields}

    输入上下文:
    {context_payload}
    """
).strip()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class PromptConfig:
    generation_system_prompt: str
    generation_user_prompt_template: str


@dataclass(frozen=True)
class RuntimeConfig:
    stage: str
    context_mode: str
    env_file: Path
    output_dir: Path
    report_dir: Path
    data_dict_path: Path
    max_factors_per_report: int
    max_samples_generation: int
    max_concurrency: int
    max_qps: float
    llm: LLMConfig
    prompts: PromptConfig


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _int_env(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None:
        return default
    return int(value)


def build_runtime_config(
    *,
    stage: str,
    context_mode: str = DEFAULT_CONTEXT_MODE,
    env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    max_factors_per_report: int = DEFAULT_MAX_FACTORS_PER_REPORT,
    max_samples_generation: int = DEFAULT_MAX_SAMPLES_GENERATION,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_qps: float = DEFAULT_MAX_QPS,
) -> RuntimeConfig:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    resolved_env_file = base_dir / DEFAULT_ENV_FILE if env_file is None else Path(env_file)
    env_values = _load_env_file(resolved_env_file)
    output_dir = base_dir / DEFAULT_OUTPUT_DIR if output_path is None else Path(output_path)
    llm = LLMConfig(
        base_url=env_values.get("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
        api_key=env_values.get("SILICONFLOW_API_KEY"),
        model=env_values.get("SILICONFLOW_MODEL", DEFAULT_SILICONFLOW_MODEL),
        timeout=_int_env(env_values, "LLM_TIMEOUT", 120),
        max_retries=_int_env(env_values, "LLM_MAX_RETRIES", 2),
    )
    prompts = PromptConfig(
        generation_system_prompt=DEFAULT_GENERATION_SYSTEM_PROMPT,
        generation_user_prompt_template=DEFAULT_GENERATION_USER_PROMPT_TEMPLATE,
    )
    return RuntimeConfig(
        stage=stage,
        context_mode=context_mode,
        env_file=resolved_env_file,
        output_dir=output_dir,
        report_dir=project_root / "研报",
        data_dict_path=base_dir / "data_dict.md",
        max_factors_per_report=max_factors_per_report,
        max_samples_generation=max_samples_generation,
        max_concurrency=max(1, max_concurrency),
        max_qps=max_qps,
        llm=llm,
        prompts=prompts,
    )
