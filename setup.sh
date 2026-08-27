#!/bin/bash
# AutoIF 项目一键环境配置脚本（AutoDL A800）
# 用法: bash setup.sh
# 预计耗时: 约30分钟（含模型下载）

set -e

export PATH=/root/miniconda3/bin:$PATH

echo "=========================================="
echo "  AutoIF 项目环境配置"
echo "=========================================="
echo "  适用平台: AutoDL A800/A100"
echo "  CUDA要求: 12.x"
echo "=========================================="

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# ========== 1. 安装 Python 依赖 ==========
echo "[1/5] 安装 Python 依赖..."

# 关键: 必须先安装正确版本的 torch（匹配CUDA 12.x）
# AutoDL 预装的 torch 可能版本过高导致 vLLM 不兼容
# 国内镜像源（AutoDL 在国内，直接访问海外源会超时）
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo "  安装 PyTorch 2.4.0+cu121（清华镜像）..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 $PIP_MIRROR -q

pip install -q $PIP_MIRROR transformers accelerate peft trl datasets
pip install -q $PIP_MIRROR openai jsonlines tqdm tenacity nltk sentencepiece
pip install -q $PIP_MIRROR modelscope huggingface-hub safetensors tiktoken
pip install -q $PIP_MIRROR tensorboard matplotlib PyYAML fire

# 安装 vLLM（必须与 torch 版本匹配）
echo "  安装 vLLM 0.5.5..."
pip install vllm==0.5.5 -q $PIP_MIRROR

# 修复 pyairports 打包 bug（0.0.1 版本缺少模块文件）
PYAIR_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/pyairports
if [ ! -f "$PYAIR_DIR/__init__.py" ] || [ ! -s "$PYAIR_DIR/__init__.py" ]; then
    echo "  修复 pyairports 模块..."
    mkdir -p "$PYAIR_DIR"
    echo 'class Airport:\n    def __init__(self, *a, **k): pass' > "$PYAIR_DIR/__init__.py"
    echo 'airports = {}' > "$PYAIR_DIR/airports.py"
fi

echo "✅ 依赖安装完成"

# ========== 2. 安装 LlamaFactory ==========
echo "[2/5] 安装 LlamaFactory..."
if [ ! -d "LlamaFactory" ]; then
    # 优先用 gitee 镜像（国内快），失败则 fallback 到 github
    git clone --depth 1 https://gitee.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || \
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory 2>/dev/null || true
fi
cd LlamaFactory && pip install -e ".[torch,metrics]" -q $PIP_MIRROR 2>/dev/null && cd ..
echo "✅ LlamaFactory 安装完成"

# ========== 3. 下载模型 ==========
echo "[3/5] 下载模型..."
mkdir -p models

# 设置 HF 镜像（AutoDL 无法直接访问 huggingface）
export HF_ENDPOINT=https://hf-mirror.com

# 教师模型: Qwen2.5-7B-Instruct (~15GB)
if [ ! -d "models/teacher" ] || [ -z "$(ls -A models/teacher 2>/dev/null)" ]; then
    echo "  下载教师模型 Qwen2.5-7B-Instruct..."
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='models/teacher')
print('教师模型下载完成')
"
else
    echo "  教师模型已存在，跳过"
fi

# 学生模型: Qwen2.5-1.5B (~3GB)
if [ ! -d "models/student" ] || [ -z "$(ls -A models/student 2>/dev/null)" ]; then
    echo "  下载学生模型 Qwen2.5-1.5B..."
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B', cache_dir='models/student')
print('学生模型下载完成')
"
else
    echo "  学生模型已存在，跳过"
fi

# NLI 模型: mDeBERTa-v3 (~2.5GB)
if [ ! -d "models/nli" ] || [ -z "$(ls -A models/nli 2>/dev/null)" ]; then
    echo "  下载 NLI 模型 mDeBERTa-v3..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
                  local_dir='models/nli')
print('NLI 模型下载完成')
"
else
    echo "  NLI 模型已存在，跳过"
fi

echo "✅ 模型下载完成"

# ========== 4. 解析模型实际路径 ==========
echo "[4/5] 配置模型路径..."

# modelscope 下载的模型在 cache_dir/org/model_name 下
# 找到实际路径并创建符号链接
TEACHER_REAL=$(find models/teacher -name "config.json" -path "*/Qwen*" | head -1 | xargs dirname 2>/dev/null || echo "")
STUDENT_REAL=$(find models/student -name "config.json" -path "*/Qwen*" | head -1 | xargs dirname 2>/dev/null || echo "")

if [ ! -z "$TEACHER_REAL" ]; then
    echo "  教师模型实际路径: $TEACHER_REAL"
fi
if [ ! -z "$STUDENT_REAL" ]; then
    echo "  学生模型实际路径: $STUDENT_REAL"
fi

# 注册数据集到 LlamaFactory
if [ -d "LlamaFactory/data" ] && [ -f "configs/llama_factory_dataset_info.json" ]; then
    python -c "
import json
lf_path = 'LlamaFactory/data/dataset_info.json'
with open(lf_path, 'r') as f:
    lf_data = json.load(f)
with open('configs/llama_factory_dataset_info.json', 'r') as f:
    custom_data = json.load(f)
lf_data.update(custom_data)
with open(lf_path, 'w') as f:
    json.dump(lf_data, f, indent=2, ensure_ascii=False)
print('  数据集注册到 LlamaFactory 完成')
"
fi

# 创建输出目录
mkdir -p output logs

# ========== 5. 环境验证 ==========
echo "[5/5] 验证环境..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f}GB')
import transformers
print(f'  Transformers: {transformers.__version__}')
try:
    import vllm
    print(f'  vLLM: {vllm.__version__}')
except: print('  vLLM: 导入失败')
try:
    import llamafactory
    print(f'  LlamaFactory: OK')
except: print('  LlamaFactory: 导入失败')
"
echo "✅ 环境验证完成"

echo ""
echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 环境工程全流程:  bash run_all.sh --domain 环境工程"
echo "  2. 指定查询集:      bash run_all.sh --domain 环境工程 --query-file data/environmental_engineering_queries.jsonl"
echo "  3. 查看领域列表:    python scripts/generate_seed_instructions.py --list"
echo ""
