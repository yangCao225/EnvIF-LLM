"""
步骤 6: 查询增强与响应生成
将指令与真实查询配对，并使用监督模型生成多个响应

输入: output/back_trans_filter.jsonl + ShareGPT 数据集 (可选)
输出: output/query_rft.jsonl (包含 gpt-answer 字段)

ShareGPT 数据路径可通过环境变量设置:
    export SHAREGPT_PATH="./data/sharegpt.json"
"""
import json
import random
import copy
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    logger, read_jsonl, write_jsonl, check_input_file,
    call_llm, call_llm_batch,
    BACK_TRANS_FILTER_PATH, QUERY_RFT_PATH,
    QUERIES_PER_INSTRUCTION, K_RESPONSE, DOMAIN_QUERY_PATH, DOMAIN_NAME
)

random.seed(0)

# ShareGPT 数据路径
SHAREGPT_PATH = os.environ.get("SHAREGPT_PATH", "./data/sharegpt.json")


def load_queries_from_sharegpt(path: str) -> list:
    """从 ShareGPT 数据集加载用户查询"""
    queries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]

        for item in data:
            # 支持 ShareGPT conversations 格式
            if 'conversations' in item:
                for msg in item['conversations']:
                    if msg.get('from') in ('human', 'user') and msg.get('value'):
                        q = msg['value'].strip()
                        if 20 < len(q) < 300:
                            queries.append(q)
                        break
            # 支持 messages 格式
            elif 'messages' in item:
                for msg in item['messages']:
                    if msg.get('role') in ('human', 'user') and msg.get('content'):
                        q = msg['content'].strip()
                        if 20 < len(q) < 300:
                            queries.append(q)
                        break

        logger.info(f"从 ShareGPT 加载 {len(queries)} 条查询")
    except FileNotFoundError:
        logger.warning(f"ShareGPT 数据集未找到: {path}")
    except Exception as e:
        logger.warning(f"加载 ShareGPT 失败: {e}")

    return queries


def load_domain_queries(path: str) -> list:
    """加载环境工程领域查询，保留 category / gold 等字段。"""
    queries = []
    if not path or not os.path.exists(path):
        return queries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = (row.get("query") or "").strip()
            if 20 < len(q) < 400:
                queries.append(row if isinstance(row, dict) else {"query": q})
    logger.info(f"从领域查询集加载 {len(queries)} 条: {path}")
    return queries


def generate_queries_with_llm(count: int = 500) -> list:
    """领域条件生成：强制环境工程问题，禁止医学/编程/金融等无关主题。"""
    logger.info(f"使用 LLM 生成 {count} 条{DOMAIN_NAME}查询...")
    prompt = f"""请生成{count}条中文环境工程用户问题，必须全部属于环境工程。
硬性要求：
1. 覆盖污水处理与水污染控制、环境监测、大气污染、固体废物、噪声、环评与清洁生产。
2. 至少30%为带具体数值的工程计算题（去除率、污染负荷、HRT、F/M等）。
3. 至少20%为工艺异常诊断题。
4. 不要生成医学、编程、金融、法律、日常生活闲聊问题。
5. 每行一个JSON：{{"query":"...","category":"wastewater_calculation|process_diagnosis|monitoring_analysis|air_pollution|solid_waste|noise_control|eia_cleaner_production","difficulty":"easy|medium|hard","required_knowledge":["..."]}}
只输出JSON行，不要编号。"""

    queries = []
    batch_size = min(40, count)
    num_batches = (count + batch_size - 1) // batch_size
    for i in range(num_batches):
        try:
            responses = call_llm(prompt, temperature=0.95, max_tokens=2048)
            for resp in responses:
                for line in resp.split("\n"):
                    line = line.strip().lstrip("0123456789.-) •")
                    row = None
                    if line.startswith("{"):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            row = None
                    if row and row.get("query"):
                        q = str(row["query"]).strip()
                        if 20 < len(q) < 400:
                            queries.append(row)
                    elif 20 < len(line) < 400:
                        queries.append({"query": line, "category": "environmental_engineering"})
        except Exception as e:
            logger.error(f"LLM 生成查询失败 (第 {i+1} 批): {e}")
    uniq = {}
    for row in queries:
        uniq[row["query"]] = row
    logger.info(f"LLM 共生成 {len(uniq)} 条不重复环境工程查询")
    return list(uniq.values())


def main():
    check_input_file(BACK_TRANS_FILTER_PATH, "步骤6-查询增强")

    # 读取过滤后的指令数据
    filter_results = read_jsonl(BACK_TRANS_FILTER_PATH)
    logger.info(f"读取 {len(filter_results)} 条过滤后的指令")

    # 优先使用环境工程领域查询集，其次 ShareGPT，最后才 LLM 领域条件生成
    domain_rows = load_domain_queries(os.environ.get("AUTOIF_QUERY_PATH", DOMAIN_QUERY_PATH))
    sharegpt_rows = [{"query": q} for q in load_queries_from_sharegpt(SHAREGPT_PATH)]
    query_rows = domain_rows + sharegpt_rows
    if len(query_rows) < QUERIES_PER_INSTRUCTION:
        logger.warning(f"查询数据不足 (当前 {len(query_rows)} 条, 需要至少 {QUERIES_PER_INSTRUCTION} 条)")
        logger.info("使用领域条件生成补充查询...")
        extra = generate_queries_with_llm(count=max(500, QUERIES_PER_INSTRUCTION * 5))
        query_rows.extend(extra)

    uniq = {}
    for row in query_rows:
        q = (row.get("query") or "").strip()
        if q:
            uniq[q] = row
    query_rows = list(uniq.values())

    if len(query_rows) < QUERIES_PER_INSTRUCTION:
        logger.error(f"查询数据严重不足 ({len(query_rows)} 条), 无法继续。")
        return

    logger.info(f"可用环境工程查询数: {len(query_rows)}")

    # ========== 阶段1: 构造查询-指令配对 ==========
    logger.info("阶段1: 构造查询-指令配对...")
    inputs = []
    for instruction_data in tqdm(filter_results, desc="构造配对"):
        sample_size = min(QUERIES_PER_INSTRUCTION, len(query_rows))
        ins_queries = random.sample(query_rows, sample_size)

        for qrow in ins_queries:
            q = qrow["query"]
            prompt = (
                f"请严格遵循格式要求回答环境工程问题。回答必须包含单位、公式或计算过程、假设条件与不确定性说明。"
                f"没有提供适用标准、地区或年份时，不得自行编造标准限值。\n"
                f"[instruction] {instruction_data['instruction']}\n"
                f"[Query] {q}"
            )
            item = copy.deepcopy(instruction_data)
            item["prompt"] = prompt
            item["query"] = q
            item["category"] = qrow.get("category")
            item["gold"] = qrow.get("gold")
            item["required_knowledge"] = qrow.get("required_knowledge")
            item["difficulty"] = qrow.get("difficulty")
            inputs.append(item)

    logger.info(f"共生成 {len(inputs)} 个查询-指令配对")

    # ========== 阶段2: 使用监督模型生成 K 个响应 ==========
    logger.info(f"阶段2: 为每个查询生成 {K_RESPONSE} 个响应...")
    prompts = [item['prompt'] for item in inputs]

    responses = call_llm_batch(
        prompts,
        system_prompt=(
            "You are an environmental engineering assistant. "
            "Follow the instruction precisely. Give units, formulas, assumptions and uncertainty. "
            "Never fabricate emission limits or monitoring data."
        ),
        temperature=0.8,
        max_tokens=2048,
        n=K_RESPONSE,
        desc="生成响应"
    )

    # 组装输出
    outputs = []
    for item, resp_list in zip(inputs, responses):
        if resp_list:
            item['gpt-answer'] = resp_list
            outputs.append(item)

    write_jsonl(QUERY_RFT_PATH, outputs)
    logger.info(f"✅ 查询增强完成! 成功生成 {len(outputs)} 条数据")


if __name__ == "__main__":
    main()
