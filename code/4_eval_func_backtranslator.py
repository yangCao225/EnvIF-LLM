"""
步骤 4: 反向翻译
将验证函数翻译回自然语言指令，用于后续的 NLI 一致性检查

输入: output/cross_validation.jsonl
输出: output/back_trans.jsonl
"""
import json
import re
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file, call_llm_batch,
    CROSS_VALIDATION_PATH, BACK_TRANS_PATH
)

# 反向翻译提示词
# 要求教师模型将 Python 验证函数“反向翻译”为自然语言指令
# 目的: 与原始指令做 NLI 一致性检查，过滤“函数和指令语义不一致”的低质量样本
BACKTRANS_PROMPT = """You are an expert in converting Python eval function code into the corresponding instruction text.
I will provide eval function code. Please strictly follow the code to convert it into the corresponding instruction text.

Here's an example:

Input functions:
[["def evaluate(response):\\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\\n    words = response.split()\\n    for word in words:\\n        if 'e' in word.lower():\\n            return False\\n    return True", 1.0]]

Output (a JSON list of instruction strings):
["Answer without using any words that contain the letter 'E'.", "Answer with words that do not contain the letter 'E'."]

Please convert the following eval functions into instructions and return ONLY a JSON list of instruction strings:
{funcs}"""


def main():
    check_input_file(CROSS_VALIDATION_PATH, "步骤4-反向翻译")

    data = read_jsonl(CROSS_VALIDATION_PATH)
    logger.info(f"开始反向翻译，共 {len(data)} 条指令")

    # 构造提示词 (每条指令取前3个验证函数)
    prompts = []
    for line in data:
        funcs = line["eval_func"][:3]
        prompt = BACKTRANS_PROMPT.format(funcs=json.dumps(funcs, ensure_ascii=False))
        prompts.append(prompt)

    # 批量调用 LLM
    responses = call_llm_batch(
        prompts,
        system_prompt="You are an expert at understanding Python code and converting it to natural language instructions.",
        temperature=0.3,
        max_tokens=1024,
        desc="反向翻译"
    )

    # 解析结果
    filter_results = []
    filter_count = 0

    for line, resp_list in zip(data, responses):
        if not resp_list:
            filter_count += 1
            continue

        response_text = resp_list[0]

        # 解析 JSON 列表
        back_instructions = None
        try:
            back_instructions = json.loads(response_text)
            if not isinstance(back_instructions, list):
                raise ValueError("Expected a list")
        except (json.JSONDecodeError, ValueError):
            # 尝试从响应中提取 JSON 列表
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                try:
                    back_instructions = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if not back_instructions or not isinstance(back_instructions, list):
            filter_count += 1
            continue

        # 确保每个元素是字符串
        back_instructions = [str(inst).strip() for inst in back_instructions if inst]

        if not back_instructions:
            filter_count += 1
            continue

        line["back_instruction"] = back_instructions
        filter_results.append(line)

    write_jsonl(BACK_TRANS_PATH, filter_results)
    logger.info(f"✅ 反向翻译完成! 成功: {len(filter_results)}, 过滤: {filter_count}")


if __name__ == "__main__":
    main()
