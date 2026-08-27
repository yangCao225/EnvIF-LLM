"""
步骤 3: 交叉验证质量控制
验证函数与测试用例互相验证，过滤低质量指令

核心逻辑:
  1. 解析 LLM 生成的验证函数和测试用例
  2. 用所有函数验证测试用例，保留有效测试用例
  3. 用有效测试用例评估每个函数的准确率
  4. 保留准确率 >= 阈值 的高质量函数

输入: output/eval_func_rft.jsonl
输出: output/cross_validation.jsonl
"""
import json
import numpy as np
import random
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    compile_eval_func, run_eval_func, parse_json_from_response,
    EVAL_FUNC_RFT_PATH, CROSS_VALIDATION_PATH,
    CROSS_VAL_THRESHOLD, MIN_EVAL_FUNCS, MIN_TEST_CASES
)

random.seed(0)


def main():
    check_input_file(EVAL_FUNC_RFT_PATH, "步骤3-交叉验证")

    results = read_jsonl(EVAL_FUNC_RFT_PATH)
    logger.info(f"开始交叉验证，共 {len(results)} 条指令")

    filter_results = []

    for result in tqdm(results, desc="交叉验证"):
        gpt_answers = result.get('gpt-answer', [])
        eval_funcs_code = []
        test_cases = []

        seed_codes = list(result.get("seed_eval_func") or [])
        try:
            from env_seed_validators import funcs_for_instruction
            seed_codes.extend(funcs_for_instruction(result.get("instruction", "")))
        except Exception:
            pass
        for code in seed_codes:
            if compile_eval_func(code):
                eval_funcs_code.append(code)

        # ===== 阶段1: 解析验证函数和测试用例 =====
        for answer in gpt_answers:
            parsed = parse_json_from_response(answer)
            if parsed is None:
                continue

            # 提取验证函数代码
            func_code = parsed.get('func', '')
            if not func_code:
                continue

            # 尝试编译验证函数
            eval_fn = compile_eval_func(func_code)
            if eval_fn is None:
                continue
            eval_funcs_code.append(func_code)

            # 提取测试用例
            cases = parsed.get('cases', [])
            for case in cases:
                try:
                    inp = str(case.get('input', ''))
                    out = case.get('output', None)
                    if out is None:
                        continue
                    # 标准化布尔值
                    if isinstance(out, str):
                        out = out.lower() in ('true', '1', 'yes')
                    else:
                        out = bool(out)
                    test_cases.append((inp, out))
                except (KeyError, TypeError):
                    continue

        # 去重
        eval_funcs_code = list(set(eval_funcs_code))
        unique_cases = list({json.dumps(c, ensure_ascii=False): c
                            for c in test_cases}.values())
        test_cases = unique_cases

        # 数量检查
        if len(eval_funcs_code) < MIN_EVAL_FUNCS or len(test_cases) < MIN_TEST_CASES:
            continue

        # ===== 阶段2: 用所有函数验证测试用例，保留有效用例 =====
        filtered_test_cases = []
        for case_input, case_output in test_cases:
            valid = False
            for func_code in eval_funcs_code:
                eval_fn = compile_eval_func(func_code)
                if eval_fn is None:
                    continue
                res = run_eval_func(eval_fn, case_input)
                if res is not None and bool(res) == case_output:
                    valid = True
                    break
            if valid:
                filtered_test_cases.append((case_input, case_output))

        if not filtered_test_cases:
            continue

        # ===== 阶段3: 用有效用例评估每个函数的准确率 =====
        scored_funcs = []
        for func_code in eval_funcs_code:
            eval_fn = compile_eval_func(func_code)
            if eval_fn is None:
                continue

            acc = []
            for case_input, case_output in filtered_test_cases:
                res = run_eval_func(eval_fn, case_input)
                if res is not None and bool(res) == case_output:
                    acc.append(1)
                else:
                    acc.append(0)

            accuracy = float(np.mean(acc)) if acc else 0.0
            scored_funcs.append((func_code, accuracy))

        # 保留高质量函数
        valid_funcs = [(f, a) for f, a in scored_funcs if a >= CROSS_VAL_THRESHOLD]
        for code in seed_codes:
            if compile_eval_func(code) and all(code != f for f, _ in valid_funcs):
                valid_funcs.append((code, 1.0))
        if not valid_funcs:
            continue

        filter_results.append({
            "instruction": result['instruction'],
            "eval_func": valid_funcs,
            "cases": filtered_test_cases
        })

    write_jsonl(CROSS_VALIDATION_PATH, filter_results)
    logger.info(f"✅ 交叉验证完成! 通过: {len(filter_results)}/{len(results)} 条指令")


if __name__ == "__main__":
    main()
