"""
步骤 9: SFT 数据构建
将过滤后的数据转换为 LLaMA Factory 的 Alpaca 格式

输入: output/query_score_filter.jsonl
输出: output/IF_sft_data.json
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, check_input_file, ensure_output_dir,
    QUERY_SCORE_FILTER_PATH, SFT_DATA_PATH
)


def main():
    check_input_file(QUERY_SCORE_FILTER_PATH, "步骤9-SFT数据构建")
    ensure_output_dir()

    data = read_jsonl(QUERY_SCORE_FILTER_PATH)
    logger.info(f"开始构建 SFT 数据，共 {len(data)} 条")

    processed_data = []
    for item in data:
        query = item.get('query', '').strip()
        instruction = item.get('instruction', '').strip()
        response = item.get('response', '').strip()
        if not response:
            ans = item.get('gpt-answer', '')
            if isinstance(ans, list) and ans:
                response = str(ans[0]).strip()
            else:
                response = str(ans).strip() if ans else ''

        # 跳过空值
        if not query or not instruction or not response:
            continue

        # 首字母大写 (安全处理空字符串)
        if len(query) > 0:
            query = query[0].upper() + query[1:]
        if len(instruction) > 0:
            instruction = instruction[0].upper() + instruction[1:]

        # 组合 query + instruction
        query_stripped = query.rstrip()
        if query_stripped.endswith('?'):
            combined = f"{query} {instruction}."
        elif query_stripped.endswith('.'):
            combined = f"{query} {instruction}."
        else:
            combined = f"{query}. {instruction}."

        new_item = {
            "instruction": combined,
            "input": "",
            "output": response,
            "history": []
        }
        if item.get("category"):
            new_item["category"] = item["category"]
        if item.get("layer_score") is not None:
            new_item["layer_score"] = item["layer_score"]
        processed_data.append(new_item)

    # 保存为 JSON (LLaMA Factory Alpaca 格式)
    with open(SFT_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ SFT 数据构建完成! 共 {len(processed_data)} 条")
    logger.info(f"   保存位置: {SFT_DATA_PATH}")


if __name__ == "__main__":
    main()
