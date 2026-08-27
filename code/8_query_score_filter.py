"""
步骤 8: 质量过滤
根据监督模型的评分过滤低质量样本

输入: output/query_rft_score.jsonl
输出: output/query_score_filter.jsonl
"""
import re
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    QUERY_RFT_SCORE_PATH, QUERY_SCORE_FILTER_PATH,
    QUALITY_SCORE_THRESHOLD
)


def main():
    check_input_file(QUERY_RFT_SCORE_PATH, "步骤8-质量过滤")

    results = read_jsonl(QUERY_RFT_SCORE_PATH)
    logger.info(f"开始质量过滤 (阈值: >{QUALITY_SCORE_THRESHOLD}), 共 {len(results)} 条")

    filter_results = []
    for result in tqdm(results, desc="质量过滤"):
        scores = []
        for gen_text in result.get('gen', []):
            # 使用贪婪匹配 + MULTILINE 标志，正确匹配 Score: 10 等多位数
            score_matches = re.findall(r'Score:\s*(\d+)\s*$', gen_text, re.MULTILINE)
            for s in score_matches:
                try:
                    scores.append(int(s))
                except ValueError:
                    continue

        avg_score = np.mean(scores) if scores else 0

        if avg_score > QUALITY_SCORE_THRESHOLD:
            result['avg_score'] = float(avg_score)
            filter_results.append(result)

    write_jsonl(QUERY_SCORE_FILTER_PATH, filter_results)
    logger.info(f"✅ 质量过滤完成! 保留: {len(filter_results)}/{len(results)} 条")


if __name__ == "__main__":
    main()
