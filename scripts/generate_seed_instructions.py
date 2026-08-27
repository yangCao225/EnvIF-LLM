#!/usr/bin/env python3
"""
领域特定种子指令生成器
根据输入的主题（如法律、美术等）自动生成种子指令
"""

import json
import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 导入扩展领域
try:
    from extended_domains import EXTENDED_DOMAINS
except ImportError:
    EXTENDED_DOMAINS = {}


# 领域特定的指令模板
DOMAIN_TEMPLATES = {
    "法律": {
        "description": "法律咨询、法规解释、案例分析",
        "seed_instructions": [
            "用不超过100字解释该法律条款",
            "列举3个相关的法律案例",
            "以律师的口吻回答，使用专业术语",
            "分点说明该行为的法律后果",
            "引用具体的法律条文支持你的观点",
            "用通俗易懂的语言解释法律概念",
            "分析该案例中的法律争议点",
            "说明该情况下的维权途径",
            "对比不同司法管辖区的相关法律",
            "评估该行为的法律风险等级（低/中/高）",
            "提供法律建议时必须包含免责声明",
            "按时间顺序说明法律程序步骤",
            "区分民事责任和刑事责任",
            "说明该法律的适用范围和例外情况",
            "用表格形式对比不同法律方案的利弊",
        ]
    },
    "美术": {
        "description": "艺术创作、美术理论、作品鉴赏",
        "seed_instructions": [
            "用艺术评论的语言描述该作品",
            "分析该画作的构图、色彩和光影",
            "说明该艺术流派的主要特征",
            "列举3位该风格的代表艺术家",
            "用不超过50字概括该艺术运动",
            "对比两种艺术风格的异同",
            "描述该技法的具体操作步骤",
            "分析该作品的象征意义和文化背景",
            "评价该艺术家的历史地位和影响",
            "用诗意的语言描绘画面意境",
            "说明该材料的特性和适用场景",
            "按时间顺序梳理艺术史发展脉络",
            "分析该作品的创新之处",
            "用表格对比不同绘画媒介的特点",
            "解释该美术术语的含义和用法",
        ]
    },
    "医疗": {
        "description": "医学知识、健康咨询、疾病科普",
        "seed_instructions": [
            "用通俗语言解释该医学术语",
            "列举该疾病的常见症状（不少于5个）",
            "说明该治疗方法的原理和注意事项",
            "对比不同治疗方案的优缺点",
            "用不超过100字科普该健康知识",
            "按严重程度分级说明该症状",
            "说明该药物的作用机制和副作用",
            "提供健康建议时必须包含就医提醒",
            "分析该疾病的风险因素",
            "用表格形式展示检查项目和正常值范围",
            "说明该疾病的预防措施",
            "解释该医学检查的目的和流程",
            "对比中医和西医的不同治疗思路",
            "说明该急救措施的操作步骤",
            "评估该症状的紧急程度（需立即就医/观察/自行处理）",
        ]
    },
    "编程": {
        "description": "代码编写、技术问答、算法讲解",
        "seed_instructions": [
            "用代码块格式展示完整的实现",
            "逐行注释解释代码逻辑",
            "说明该算法的时间复杂度和空间复杂度",
            "列举至少3个使用场景",
            "对比不同实现方式的性能差异",
            "用不超过50字总结该技术的核心思想",
            "提供可运行的完整代码示例",
            "说明该方法的参数类型和返回值",
            "指出代码中的潜在问题和优化建议",
            "用表格对比不同技术方案",
            "按步骤说明该功能的实现思路",
            "解释该设计模式的应用场景",
            "提供单元测试用例",
            "说明该技术的浏览器兼容性",
            "用流程图描述算法执行过程",
        ]
    },
    "金融": {
        "description": "投资理财、金融知识、市场分析",
        "seed_instructions": [
            "用不超过100字解释该金融概念",
            "分析该投资产品的风险等级",
            "列举该理财方式的优缺点",
            "对比不同投资策略的适用人群",
            "说明该金融指标的计算方法",
            "用表格展示不同资产配置方案",
            "分析该市场趋势的影响因素",
            "提供投资建议时必须包含风险提示",
            "说明该金融工具的运作机制",
            "评估该投资的预期收益率",
            "解释该经济现象的成因",
            "对比国内外市场的差异",
            "说明该金融政策的影响",
            "按风险从低到高排序投资产品",
            "分析该公司的财务状况",
        ]
    },
    "教育": {
        "description": "教学方法、学习指导、知识讲解",
        "seed_instructions": [
            "用简单易懂的语言解释该概念",
            "设计3个练习题帮助理解",
            "用类比的方式说明该原理",
            "列举该知识点的实际应用",
            "按难度递进的方式组织内容",
            "用不超过50字总结核心要点",
            "提供记忆口诀或助记方法",
            "对比易混淆的概念",
            "用图表辅助说明",
            "设计互动问题引导思考",
            "说明该学习方法的适用场景",
            "分析常见错误和纠正方法",
            "提供拓展阅读建议",
            "用故事化的方式讲解知识",
            "评估学习效果的检验方法",
        ]
    },
    "通用": {
        "description": "通用格式指令，适用于各种场景",
        "seed_instructions": [
            "回答不超过25个字",
            "限制在3句话以内回答",
            "使用不超过15个不重复的词回答",
            "用单个段落回答，不要换行",
            "回答必须恰好包含20个字",
            "每句话只用5个字",
            "将回答限制在一个段落内",
            "回答恰好20个字，不多不少",
            "用一句话回答，恰好100个字",
            "回答不超过50字且每个字都有意义",
            "整个回答不超过30字",
            "用5个要点回答，每个要点单独一行",
            "用编号列表的形式组织回答",
            "每句话的开头用不同的字",
            "只使用常用的2000个汉字",
            "回答中必须包含3个反问句",
            "用一句谚语或俗语结束回答",
            "用对话的形式回答",
            "用古诗词的格式回答",
            "回答必须押韵",
            "用排比句式回答",
            "每句话不超过10个字",
            "用总分总的结构组织回答",
            "回答中不能出现标点符号",
            "用第一人称回答",
            "用第三人称客观描述",
            "回答必须包含具体的数字",
            "用比喻的方式解释",
            "用举例的方式说明",
            "分3个层次递进说明",
        ]
    }
}

# 合并扩展领域
DOMAIN_TEMPLATES.update(EXTENDED_DOMAINS)


def generate_seed_instructions(domain, output_file=None, count=30, use_llm=False):
    """
    生成领域特定的种子指令
    
    Args:
        domain: 领域名称（法律、美术、医疗等）
        output_file: 输出文件路径
        count: 生成数量
        use_llm: 是否使用LLM生成（需要配置API）
    """
    
    if domain not in DOMAIN_TEMPLATES and not use_llm:
        print(f"❌ 未找到领域 '{domain}' 的模板")
        print(f"✅ 可用领域: {', '.join(DOMAIN_TEMPLATES.keys())}")
        print(f"💡 提示: 使用 --use-llm 选项让AI生成自定义领域的指令")
        return
    
    instructions = []
    
    if use_llm:
        # 使用LLM生成自定义领域的指令
        instructions = generate_with_llm(domain, count)
    else:
        # 使用预定义模板
        template = DOMAIN_TEMPLATES[domain]
        base_instructions = [x for x in template["seed_instructions"] if x]
        env_domains = {"环境工程", "污水处理", "环境监测"}
        if domain in env_domains:
            instructions = base_instructions[:count]
        elif count > len(base_instructions):
            general_instructions = DOMAIN_TEMPLATES["通用"]["seed_instructions"]
            instructions = base_instructions + general_instructions[:count - len(base_instructions)]
        else:
            instructions = base_instructions[:count]
    
    # 保存到文件
    if output_file is None:
        output_file = f"sample_data/seed_instruction_{domain}.txt"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for instruction in instructions:
            f.write(instruction + '\n')
    
    print(f"✅ 已生成 {len(instructions)} 条种子指令")
    print(f"📁 保存位置: {output_path}")
    print(f"\n📋 前5条示例:")
    for i, inst in enumerate(instructions[:5], 1):
        print(f"  {i}. {inst}")


def generate_with_llm(domain, count):
    """
    使用LLM生成自定义领域的种子指令
    """
    prompt = f"""你是一个指令生成专家。请为"{domain}"领域生成{count}条种子指令。

要求：
1. 指令应该是关于响应格式的要求，而不是内容风格
2. 指令必须可以用Python函数验证（例如：字数限制、句数限制、格式要求等）
3. 指令应该适用于{domain}领域的问答场景
4. 每条指令一行，不要编号

好的指令示例：
- 用不超过100字解释该概念
- 列举至少3个相关案例
- 以专业术语回答，包含领域特定词汇
- 分点说明，每点不超过50字
- 用表格形式对比不同方案

不好的指令示例（不要生成这类）：
- 用诗意的语言描述（风格类，难以验证）
- 使用隐喻手法（风格类）
- 翻译成其他语言（翻译类）

请直接输出{count}条指令，每行一条："""

    print(f"🤖 正在使用LLM生成 {domain} 领域的种子指令...")
    print(f"💡 提示: 需要配置 OPENAI_API_KEY 环境变量")
    
    # 这里需要调用LLM API
    # 示例使用OpenAI API
    try:
        from openai import OpenAI
        
        # 支持 vLLM 本地服务和 OpenAI API
        api_base = os.getenv("SUPERVISOR_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        api_key = os.getenv("SUPERVISOR_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        model = os.getenv("SUPERVISOR_MODEL", "gpt-4")
        
        client = OpenAI(base_url=api_base, api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的指令生成专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        instructions = [line.strip().lstrip('- ').lstrip('• ') 
                       for line in content.split('\n') 
                       if line.strip() and not line.strip().startswith('#')]
        
        return instructions[:count]
        
    except ImportError:
        print("❌ 未安装 openai 库，请运行: pip install openai")
        return []
    except Exception as e:
        print(f"❌ LLM调用失败: {e}")
        return []


def list_domains():
    """列出所有可用的领域"""
    print("📚 可用领域模板:\n")
    for domain, info in DOMAIN_TEMPLATES.items():
        print(f"  🏷️  {domain}")
        print(f"     {info['description']}")
        print(f"     示例: {info['seed_instructions'][0]}")
        print()


def interactive_mode():
    """交互式模式"""
    print("🎯 种子指令生成器 - 交互模式\n")
    
    # 显示可用领域
    print("📚 可用领域分类:\n")
    
    categories = {
        "环境工程（本项目主线）": ["环境工程", "污水处理", "环境监测"],
        "基础科学": ["数学", "物理", "化学", "生物", "天文", "地理"],
        "传统工程": ["土木工程", "机械工程", "电子工程", "化工", "材料科学", "能源工程"],
        "人文社科": ["文学", "历史", "哲学", "新闻传播", "社会学", "心理学"],
        "经济管理": ["工商管理", "会计", "公共管理", "电子商务", "金融"],
        "应用领域": ["法律", "医疗", "教育", "编程"],
        "艺术体育": ["美术", "音乐", "体育"],
        "其他": ["农学", "通用"],
    }
    
    for category, domains in categories.items():
        print(f"  🏷️  {category}: {', '.join(domains)}")
    
    print("\n" + "="*60)
    
    # 选择领域
    domain = input("\n请输入领域名称（或输入'list'查看详细说明）: ").strip()
    
    if domain.lower() == 'list':
        list_domains()
        domain = input("\n请输入领域名称: ").strip()
    
    if domain not in DOMAIN_TEMPLATES:
        use_llm_choice = input(f"\n未找到领域 '{domain}'，是否使用AI生成？(y/n): ").strip().lower()
        use_llm = use_llm_choice == 'y'
        if not use_llm:
            print("❌ 已取消")
            return
    else:
        use_llm = False
    
    # 选择数量
    count_input = input(f"\n生成数量 [默认30]: ").strip()
    count = int(count_input) if count_input else 30
    
    # 输出文件
    output_input = input(f"\n输出文件 [默认: sample_data/seed_instruction_{domain}.txt]: ").strip()
    output_file = output_input if output_input else None
    
    # 生成
    print("\n" + "="*60)
    generate_seed_instructions(domain, output_file, count, use_llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="领域特定种子指令生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成法律领域的30条种子指令
  python generate_seed_instructions.py --domain 法律
  
  # 生成数学领域的50条种子指令
  python generate_seed_instructions.py --domain 数学 --count 50
  
  # 生成物理领域的指令
  python generate_seed_instructions.py --domain 物理
  
  # 使用AI生成自定义领域的指令
  python generate_seed_instructions.py --domain 建筑设计 --use-llm
  
  # 交互式模式（推荐）
  python generate_seed_instructions.py --interactive
  
  # 列出所有可用领域
  python generate_seed_instructions.py --list

可用领域分类:
  环境工程: 环境工程、污水处理、环境监测
  基础科学: 数学、物理、化学、生物、天文、地理
  传统工程: 土木工程、机械工程、电子工程、化工、材料科学、能源工程
  人文社科: 文学、历史、哲学、新闻传播、社会学、心理学
  经济管理: 工商管理、会计、公共管理、电子商务、金融
  应用领域: 法律、医疗、教育、编程
  艺术体育: 美术、音乐、体育
  其他: 农学、通用
        """
    )
    
    parser.add_argument("--domain", type=str, help="领域名称（如：数学、物理、法律、医疗等，详见--list）")
    parser.add_argument("--count", type=int, default=30, help="生成数量（默认30）")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--use-llm", action="store_true", help="使用LLM生成自定义领域指令")
    parser.add_argument("--list", action="store_true", help="列出所有可用领域")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式模式")
    
    args = parser.parse_args()
    
    if args.list:
        list_domains()
    elif args.interactive:
        interactive_mode()
    elif args.domain:
        generate_seed_instructions(args.domain, args.output, args.count, args.use_llm)
    else:
        parser.print_help()
