"""
步骤 7: 查询验证与质量评分
使用验证函数过滤不合规响应，并使用监督模型评估响应质量

输入: output/query_rft.jsonl
输出: output/query_rft_score.jsonl
"""
import json
import re
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    compile_eval_func, run_eval_func, call_llm_batch,
    QUERY_RFT_PATH, QUERY_RFT_SCORE_PATH, CALC_REL_TOL
)

# 质量评分提示词
# 要求教师模型从 0-10 评分: 响应是否既遵循了指令格式又回答了用户查询
# 输出格式: 详细分析 + 末尾 "Score: {score}"
SCORING_PROMPT = """You are an expert that is good at judging whether a response is following the instruction and query.
[Instruction] {instruction}
[Query] {query}
[Response] {response}
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""


def main():
    check_input_file(QUERY_RFT_PATH, "步骤7-查询验证")

    results = read_jsonl(QUERY_RFT_PATH)
    logger.info(f"开始查询验证，共 {len(results)} 条")

    # ========== 阶段1: 使用验证函数过滤响应 ==========
    logger.info("阶段1: 验证函数过滤...")
    filter_samples = []

    for result in tqdm(results, desc="验证过滤"):
        # 编译验证函数
        eval_funcs = []
        for func_code, score in result.get('eval_func', []):
            fn = compile_eval_func(func_code)
            if fn is not None:
                eval_funcs.append(fn)

        if not eval_funcs:
            continue

        # 验证每个响应
        for response in result.get('gpt-answer', []):
            acc = []
            for eval_fn in eval_funcs:
                res = run_eval_func(eval_fn, response)
                if res is not None:
                    try:
                        acc.append(int(bool(res)))
                    except (ValueError, TypeError):
                        continue

            accuracy = np.mean(acc) if acc else 0

            # 三层验证：格式 + 计算 + 专业依据（有 gold 时启用）
            try:
                from env_validators import evaluate_response, instruction_constraints, score_response
                fmt_cons = instruction_constraints(result.get("instruction", ""))
                layered = evaluate_response(
                    response,
                    query=result.get("query", ""),
                    gold=result.get("gold") or {},
                    format_constraints=fmt_cons,
                    rel_tol=CALC_REL_TOL,
                )
                layer_score = score_response(layered)
            except Exception:
                layered, layer_score = None, None

            # 三层验证分数写入样本，不因计算层失败直接丢弃（教师模型也可能算错，留给 DPO 做偏好）
            if accuracy > 0:
                query = result.get('query', '')
                if not query:
                    match = re.findall(r'\[Query\]\s*(.*)$', result.get('prompt', ''), re.DOTALL)
                    query = match[0].strip() if match else ''

                if query:
                    sample = {
                        'instruction': result['instruction'],
                        'query': query,
                        'response': response,
                        'format_acc': float(accuracy),
                    }
                    if result.get('gold'):
                        sample['gold'] = result['gold']
                    if result.get('category'):
                        sample['category'] = result['category']
                    if layered is not None:
                        sample['layer_eval'] = layered
                        sample['layer_score'] = layer_score
                        if layered.get("error_types"):
                            sample['error_types'] = layered["error_types"]
                    filter_samples.append(sample)

    # 去重
    unique_samples = list({json.dumps(s, ensure_ascii=False): s
                          for s in filter_samples}.values())
    logger.info(f"验证过滤后: {len(unique_samples)} 条 (过滤前总响应数: {sum(len(r.get('gpt-answer', [])) for r in results)})")

    if not unique_samples:
        logger.warning("没有通过验证的样本，请检查前序步骤的数据质量")
        return

    # ========== 阶段2: 使用监督模型评估质量 ==========
    logger.info("阶段2: 质量评分...")

    scoring_prompts = []
    for sample in unique_samples:
        prompt = SCORING_PROMPT.format(
            instruction=sample['instruction'],
            query=sample['query'],
            response=sample['response']
        )
        sample['prompt'] = prompt
        scoring_prompts.append(prompt)

    # 批量调用 LLM 评分
    score_responses = call_llm_batch(
        scoring_prompts,
        system_prompt="You are an expert evaluator for instruction-following quality.",
        temperature=0.1,
        max_tokens=1024,
        n=1,
        desc="质量评分"
    )

    # 组装结果
    scored_data = []
    for sample, resp_list in zip(unique_samples, score_responses):
        if resp_list:
            sample['gen'] = resp_list
            scored_data.append(sample)

    write_jsonl(QUERY_RFT_SCORE_PATH, scored_data)
    logger.info(f"✅ 质量评分完成! {len(scored_data)} 条数据已评分")


if __name__ == "__main__":
    main()
