"""
DPO 步骤 1: 响应评分
使用验证函数评估每个查询的所有响应，为每个响应计算准确率分数

核心逻辑:
  对步骤6生成的每个响应，用所有验证函数评分 (0~1)
  分数高 = 较好地遵循了指令格式约束 (正样本)
  分数低 = 未遵循指令格式约束 (负样本)
  后续 DPO 步骤2 将用这些分数构建偏好对

输入: output/query_rft.jsonl (步骤6的输出，包含多个 RFT 响应)
输出: output/dpo_eval_score_results.jsonl
"""
import json
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    compile_eval_func, run_eval_func,
    QUERY_RFT_PATH, DPO_EVAL_SCORE_PATH
)


def main():
    check_input_file(QUERY_RFT_PATH, "DPO步骤1-响应评分")

    results = read_jsonl(QUERY_RFT_PATH)
    logger.info(f"开始 DPO 响应评分，共 {len(results)} 条")

    filter_samples = []
    total_responses = 0
    scored_count = 0

    for result in tqdm(results, desc="DPO 评分"):
        # 编译验证函数
        eval_funcs = []
        for func_code, score in result.get('eval_func', []):
            fn = compile_eval_func(func_code)
            if fn is not None:
                eval_funcs.append(fn)

        if not eval_funcs:
            continue

        # 评估每个响应
        scored_responses = []
        for response in result.get('gpt-answer', []):
            total_responses += 1
            acc = []
            for eval_fn in eval_funcs:
                res = run_eval_func(eval_fn, response)
                if res is not None:
                    try:
                        acc.append(int(bool(res)))
                    except (ValueError, TypeError):
                        continue

            accuracy = float(np.mean(acc)) if acc else 0.0
            scored_responses.append([response, accuracy])
            scored_count += 1

        if scored_responses:
            query_text = result.get('query', '')
            instruction_text = result.get('instruction', '')
            full_prompt = (
                f"{query_text}\n\n【格式要求】{instruction_text}"
                if query_text or instruction_text
                else result.get('prompt', '')
            )
            sample = {
                'query': query_text,
                'instruction': instruction_text,
                'prompt': full_prompt,
                'response': scored_responses,
            }
            if result.get('gold'):
                sample['gold'] = result['gold']
            if result.get('category'):
                sample['category'] = result['category']
            filter_samples.append(sample)

    # 去重
    unique_samples = list({json.dumps(s, ensure_ascii=False): s
                          for s in filter_samples}.values())

    write_jsonl(DPO_EVAL_SCORE_PATH, unique_samples)
    logger.info(f"✅ DPO 评分完成! {len(unique_samples)} 条数据, "
                f"共评估 {scored_count}/{total_responses} 个响应")


if __name__ == "__main__":
    main()
