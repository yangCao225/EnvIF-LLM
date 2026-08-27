"""
AutoIF Pipeline 工具模块
提供统一的 LLM 调用接口、路径管理和通用工具函数

使用前请设置环境变量 (在 AutoDL 上使用 vLLM 部署教师模型):
    export SUPERVISOR_API_BASE="http://localhost:8000/v1"   # vLLM 服务地址
    export SUPERVISOR_API_KEY="EMPTY"                        # API Key (vLLM 默认)
    export SUPERVISOR_MODEL="Qwen2.5-7B-Instruct"           # 模型名称 (与 vLLM --served-model-name 一致)
"""

import os
import json
import re
import time
import signal
import logging
from typing import List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("autoif")

# ========== 路径与超参数（优先读取 pipeline_config.yaml） ==========
from config_loader import get as _cfg, resolve_path as _resolve_path, project_root as _project_root

OUTPUT_DIR = os.path.join(_project_root(), "output")
SAMPLE_DIR = os.path.join(_project_root(), "sample_data")

_default_seed = os.path.join(SAMPLE_DIR, "seed_instruction_环境工程.txt")
SEED_INSTRUCTION_PATH = os.environ.get(
    "AUTOIF_SEED_PATH",
    _resolve_path(_cfg("data.seed_instructions"), _default_seed),
)
DOMAIN_QUERY_PATH = os.environ.get(
    "AUTOIF_QUERY_PATH",
    _resolve_path(_cfg("data.query_file"), os.path.join(_project_root(), "data", "environmental_engineering_queries.jsonl")),
)
DOMAIN_TEST_PATH = os.environ.get(
    "AUTOIF_TEST_PATH",
    _resolve_path(_cfg("evaluation.test_set"), os.path.join(_project_root(), "data", "environmental_engineering_test.jsonl")),
)
DOMAIN_NAME = os.environ.get("AUTOIF_DOMAIN", _cfg("data.domain", "环境工程"))

AUGMENT_INSTRUCTIONS_PATH = os.path.join(OUTPUT_DIR, "augment_instructions.txt")
EVAL_FUNC_RFT_PATH = os.path.join(OUTPUT_DIR, "eval_func_rft.jsonl")
CROSS_VALIDATION_PATH = os.path.join(OUTPUT_DIR, "cross_validation.jsonl")
BACK_TRANS_PATH = os.path.join(OUTPUT_DIR, "back_trans.jsonl")
BACK_TRANS_FILTER_PATH = os.path.join(OUTPUT_DIR, "back_trans_filter.jsonl")
QUERY_RFT_PATH = os.path.join(OUTPUT_DIR, "query_rft.jsonl")
QUERY_RFT_SCORE_PATH = os.path.join(OUTPUT_DIR, "query_rft_score.jsonl")
QUERY_SCORE_FILTER_PATH = os.path.join(OUTPUT_DIR, "query_score_filter.jsonl")
SFT_DATA_PATH = os.path.join(OUTPUT_DIR, "IF_sft_data.json")
DPO_EVAL_SCORE_PATH = os.path.join(OUTPUT_DIR, "dpo_eval_score_results.jsonl")
DPO_PAIRS_PATH = os.path.join(OUTPUT_DIR, "dpo_pairs.jsonl")

K_AUGMENT = int(_cfg("data.augmentation.k_augment", 3))
K_VERIFICATION = int(_cfg("data.validation.k_verification", 5))
K_RESPONSE = int(_cfg("data.validation.k_response", 5))
CROSS_VAL_THRESHOLD = float(_cfg("data.quality_thresholds.cross_validation_acc", 0.8))
MIN_EVAL_FUNCS = int(_cfg("data.quality_thresholds.min_functions", 3))
MIN_TEST_CASES = int(_cfg("data.quality_thresholds.min_test_cases", 10))
QUALITY_SCORE_THRESHOLD = float(_cfg("data.quality_thresholds.quality_score", 8))
QUERIES_PER_INSTRUCTION = int(_cfg("data.query.queries_per_instruction", 16))
DPO_POSITIVE_THRESHOLD = float(_cfg("data.dpo.positive_threshold", 0.5))
EXEC_TIMEOUT = int(_cfg("data.validation.exec_timeout", _cfg("misc.timeout", 5)))
LLM_MAX_WORKERS = int(_cfg("data.validation.llm_max_workers", 8))
CALC_REL_TOL = float(_cfg("data.validation.calc_rel_tol", 0.02))


# ========== 文件操作 ==========

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_input_file(path: str, step_name: str):
    """检查输入文件是否存在，不存在则抛出友好错误"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"[{step_name}] 输入文件不存在: {path}\n"
            f"请先运行上一步生成该文件。\n"
            f"{'='*60}"
        )


def write_jsonl(path: str, data: list):
    """写入 JSONL 文件"""
    ensure_output_dir()
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"已写入 {len(data)} 条数据到 {path}")


def read_jsonl(path: str) -> list:
    """读取 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"从 {path} 读取 {len(data)} 条数据")
    return data


# ========== LLM 调用 ==========

def get_llm_client():
    """获取 OpenAI 兼容的 LLM 客户端 (支持 vLLM 本地服务)"""
    # 为了线程安全，每次调用创建独立 client
    from openai import OpenAI
    api_base = os.environ.get("SUPERVISOR_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("SUPERVISOR_API_KEY", "EMPTY")
    client = OpenAI(base_url=api_base, api_key=api_key)
    return client


def get_model_name():
    """获取监督模型名称"""
    return os.environ.get("SUPERVISOR_MODEL", "default")


def call_llm(prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2048,
             n: int = 1) -> List[str]:
    """
    调用 LLM 生成响应

    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词 (可选)
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        n: 每次请求生成的响应数 (用于 RFT 多次采样)

    Returns:
        响应文本列表 (长度为 n)
    """
    client = get_llm_client()
    model = get_model_name()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            err_msg = str(e).lower()
            # 若服务不支持 n > 1，回退为多次单独调用
            if n > 1 and ("n" in err_msg or "not support" in err_msg):
                logger.info(f"模型不支持 n={n}，回退为 {n} 次单独调用")
                results = []
                for _ in range(n):
                    r = call_llm(prompt, system_prompt, temperature, max_tokens, n=1)
                    results.extend(r)
                return results

            logger.warning(f"LLM 调用失败 (第 {attempt+1}/{max_retries} 次): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM 调用失败，已重试 {max_retries} 次")


def call_llm_batch(prompts: List[str], system_prompt: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 2048,
                   n: int = 1, max_workers: int = None,
                   desc: str = "LLM 批量调用") -> List[List[str]]:
    """
    批量调用 LLM (多线程并发)

    Args:
        prompts: 提示词列表
        system_prompt: 系统提示词 (可选)
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        n: 每个提示词生成的响应数
        max_workers: 最大并发数 (默认使用 LLM_MAX_WORKERS)
        desc: 进度条描述

    Returns:
        响应列表，results[i] 是第 i 个提示词的 n 条响应
    """
    if max_workers is None:
        max_workers = LLM_MAX_WORKERS

    results = [None] * len(prompts)
    failed_indices = []

    def _call_single(idx: int, prompt: str):
        return idx, call_llm(prompt, system_prompt, temperature, max_tokens, n)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_single, i, p): i for i, p in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts), desc=desc):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                logger.error(f"第 {idx} 个提示词调用失败: {e}")
                results[idx] = []
                failed_indices.append(idx)

    if failed_indices:
        logger.warning(f"共有 {len(failed_indices)}/{len(prompts)} 个调用失败")

    return results


# ========== 验证函数工具 ==========

def compile_eval_func(func_code: str) -> Optional[Callable]:
    """
    编译验证函数代码，返回可调用的 evaluate 函数对象

    Args:
        func_code: Python 函数代码字符串

    Returns:
        evaluate 函数对象，或 None (如果编译失败)
    """
    # 处理转义字符
    if '\\n' in func_code:
        func_code = func_code.replace('\\n', '\n')
    if '\\t' in func_code:
        func_code = func_code.replace('\\t', '\t')
    func_code = func_code.strip()

    # 过滤危险代码行
    safe_lines = []
    for line in func_code.split('\n'):
        line_lower = line.strip().lower()
        if any(kw in line_lower for kw in [
            'os.system', 'subprocess', 'shutil.rmtree', '__import__',
            'download', 'requests.', 'urllib', 'socket.', 'os.remove'
        ]):
            continue
        safe_lines.append(line)
    func_code = '\n'.join(safe_lines)

    local_vars = {}
    try:
        exec(func_code, {"__builtins__": __builtins__}, local_vars)
    except Exception:
        return None

    return local_vars.get('evaluate', None)


def run_eval_func(eval_func: Callable, input_str: str, timeout: int = None) -> Any:
    """
    运行验证函数 (带超时保护)

    - Linux: 使用 signal.SIGALRM
    - Windows: 使用线程超时控制
    """
    if timeout is None:
        timeout = EXEC_TIMEOUT

    # Windows 环境不支持 SIGALRM
    if os.name == "nt":
        import threading

        result = [None]
        error = [None]

        def _target():
            try:
                result[0] = eval_func(input_str)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return None
        if error[0] is not None:
            return None
        return result[0]

    def timeout_handler(signum, frame):
        raise TimeoutError("函数执行超时")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        result = eval_func(input_str)
    except Exception:
        result = None
    finally:
        signal.alarm(0)

    return result


def parse_json_from_response(response: str) -> Optional[dict]:
    """
    从 LLM 响应中解析 JSON
    按优先级尝试: ```json``` 代码块 → ``` 代码块 → 直接解析 → 正则提取
    """
    # 1. 尝试从 ```json...``` 代码块中提取
    json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_blocks:
        try:
            return json.loads(json_blocks[0])
        except json.JSONDecodeError:
            pass

    # 2. 尝试从 ```...``` 代码块中提取
    code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    for block in code_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # 3. 尝试直接解析整个响应
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # 4. 尝试正则提取 JSON 对象
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None
