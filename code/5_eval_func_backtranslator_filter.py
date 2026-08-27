"""
步骤 5: NLI 一致性过滤
使用自然语言推断 (NLI) 模型检测原始指令与反向翻译指令的一致性
过滤掉包含矛盾的样本

输入: output/back_trans.jsonl
输出: output/back_trans_filter.jsonl

NLI 模型路径可通过环境变量设置:
    export NLI_MODEL_PATH="./models/mDeBERTa-v3-base-xnli"
"""
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    BACK_TRANS_PATH, BACK_TRANS_FILTER_PATH
)

# NLI 模型路径 (可通过环境变量设置)
DEFAULT_NLI_CANDIDATES = [
    os.environ.get("NLI_MODEL_PATH"),
    "./models/mDeBERTa-v3-base-xnli",
    "./models/mdeberta",
    "./models/mdeberta/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    os.path.expanduser("~/.cache/modelscope/mdeberta"),
]


def resolve_nli_model_path() -> str:
    """解析可用的 NLI 模型路径"""
    for path in DEFAULT_NLI_CANDIDATES:
        if not path:
            continue
        if os.path.isdir(path):
            # 目录中包含 config.json 即认为有效
            if os.path.exists(os.path.join(path, "config.json")):
                return path
            # 尝试在子目录中寻找模型
            for root, _, files in os.walk(path):
                if "config.json" in files:
                    return root
    raise FileNotFoundError(
        "未找到 NLI 模型，请先下载并设置环境变量 NLI_MODEL_PATH，或运行:\n"
        "bash setup.sh"
    )


def main():
    check_input_file(BACK_TRANS_PATH, "步骤5-NLI过滤")

    # ========== 加载 NLI 模型 (只加载一次!) ==========
    try:
        nli_model_path = resolve_nli_model_path()
        logger.info(f"加载 NLI 模型: {nli_model_path}")

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(device)
        model.eval()

        label_names = ["entailment", "neutral", "contradiction"]
        logger.info("NLI 模型加载完成")
    except Exception as e:
        logger.error(f"加载 NLI 模型失败: {e}")
        raise

    # ========== 读取数据 ==========
    data = read_jsonl(BACK_TRANS_PATH)
    logger.info(f"开始 NLI 一致性过滤，共 {len(data)} 条")

    # ========== 逐条处理 ==========
    filter_results = []
    filter_count = 0

    for line in tqdm(data, desc="NLI 过滤"):
        back_instructions = line.get("back_instruction", [])
        ori_ins = line["instruction"]

        nli_scores = []
        for back_ins in back_instructions[:3]:
            with torch.no_grad():
                inputs = tokenizer(
                    ori_ins, back_ins,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                output = model(**inputs)
                prediction = torch.softmax(output.logits[0], dim=-1).tolist()
                score_dict = {
                    name: round(float(pred) * 100, 1)
                    for pred, name in zip(prediction, label_names)
                }
                max_label = max(score_dict, key=score_dict.get)
                nli_scores.append(max_label)

        line["nli_scores"] = nli_scores

        # 如果存在矛盾，过滤掉
        if "contradiction" in nli_scores:
            filter_count += 1
            continue

        filter_results.append(line)

    write_jsonl(BACK_TRANS_FILTER_PATH, filter_results)
    logger.info(f"✅ NLI 过滤完成! 保留: {len(filter_results)}, 过滤: {filter_count}")


if __name__ == "__main__":
    main()
