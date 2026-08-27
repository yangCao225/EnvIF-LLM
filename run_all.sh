#!/bin/bash
# 面向污水处理与环境监测的 AutoIF 环境工程指令遵循优化
# 前置条件: 已运行 bash setup.sh
# 用法:
#   bash run_all.sh                              # 默认环境工程
#   bash run_all.sh --domain 环境工程
#   bash run_all.sh --domain 污水处理 --query-file data/environmental_engineering_queries.jsonl
# 后台运行: nohup bash run_all.sh > run.log 2>&1 &

set -e

export PATH=/root/miniconda3/bin:$PATH

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

DOMAIN="环境工程"
QUERY_FILE=""
TEST_FILE=""
SKIP_TRAIN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2 ;;
        --query-file) QUERY_FILE="$2"; shift 2 ;;
        --test-file) TEST_FILE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        *) shift ;;
    esac
done

read_cfg() {
    python code/config_loader.py "$1"
}

if [ -z "$QUERY_FILE" ]; then
    QUERY_FILE=$(read_cfg data.query_file)
    QUERY_FILE=${QUERY_FILE:-data/environmental_engineering_queries.jsonl}
fi
if [ -z "$TEST_FILE" ]; then
    TEST_FILE=$(read_cfg evaluation.test_set)
    TEST_FILE=${TEST_FILE:-data/environmental_engineering_test.jsonl}
fi

SFT_LR=$(read_cfg training.sft.learning_rate)
DPO_LR=$(read_cfg training.dpo.learning_rate)
SFT_LR=${SFT_LR:-5e-5}
DPO_LR=${DPO_LR:-5e-6}
SFT_EPOCHS=$(read_cfg training.sft.epochs)
DPO_EPOCHS=$(read_cfg training.dpo.epochs)
SFT_EPOCHS=${SFT_EPOCHS:-3}
DPO_EPOCHS=${DPO_EPOCHS:-2}
SFT_BS=$(read_cfg training.sft.batch_size)
DPO_BS=$(read_cfg training.dpo.batch_size)
SFT_BS=${SFT_BS:-4}
DPO_BS=${DPO_BS:-2}
SFT_GA=$(read_cfg training.sft.gradient_accumulation)
DPO_GA=$(read_cfg training.dpo.gradient_accumulation)
SFT_GA=${SFT_GA:-4}
DPO_GA=${DPO_GA:-4}
PREF_BETA=$(read_cfg training.dpo.pref_beta)
PREF_BETA=${PREF_BETA:-0.1}

echo ""
echo "============================================"
echo "  AutoIF 环境工程指令遵循优化"
echo "  领域: $DOMAIN"
echo "  查询集: $QUERY_FILE"
echo "  SFT lr: $SFT_LR | DPO lr: $DPO_LR"
echo "  开始: $(date)"
echo "============================================"

TEACHER_PATH=$(find models/teacher -name "config.json" -path "*/Qwen*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "models/teacher")
STUDENT_PATH=$(find models/student -name "config.json" -path "*/Qwen*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "models/student")

echo "教师模型: $TEACHER_PATH"
echo "学生模型: $STUDENT_PATH"

export SUPERVISOR_API_BASE="http://localhost:8000/v1"
export SUPERVISOR_API_KEY="EMPTY"
export SUPERVISOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
export NLI_MODEL_PATH="./models/nli"
export HF_ENDPOINT="https://hf-mirror.com"
export AUTOIF_DOMAIN="$DOMAIN"
export AUTOIF_QUERY_PATH="$QUERY_FILE"
export AUTOIF_TEST_PATH="$TEST_FILE"

mkdir -p output logs eval_results

# 使用独立领域种子文件 sample_data/seed_instruction_环境工程.txt
if [ "$DOMAIN" = "通用" ]; then
    SEED_FILE="sample_data/seed_instruction_环境工程.txt"
else
    SEED_FILE="sample_data/seed_instruction_${DOMAIN}.txt"
    if [ ! -f "$SEED_FILE" ]; then
        echo "生成 $DOMAIN 领域种子指令..."
        python scripts/generate_seed_instructions.py --domain "$DOMAIN" --count 40 --output "$SEED_FILE"
    fi
fi
if [ ! -f "$SEED_FILE" ]; then
    echo "❌ 找不到种子文件: $SEED_FILE"
    exit 1
fi
export AUTOIF_SEED_PATH="$SEED_FILE"
echo "✅ 使用独立种子文件 $SEED_FILE（$(wc -l < "$SEED_FILE") 条），未覆盖原始种子"

cleanup() {
    if [ ! -z "$VLLM_PID" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm.entrypoints" 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "============================================"
echo "  阶段1: 启动 vLLM 教师模型"
echo "============================================"

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM 服务已在运行"
else
    echo "启动 vLLM 服务..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$TEACHER_PATH" \
        --served-model-name Qwen/Qwen2.5-7B-Instruct \
        --port 8000 \
        --trust-remote-code \
        --gpu-memory-utilization 0.5 \
        --max-model-len 4096 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
    echo "等待 vLLM 服务就绪..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ vLLM 服务就绪（等待 ${i}s）"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "❌ vLLM 启动超时"
            exit 1
        fi
        sleep 5
    done
fi

echo ""
echo "============================================"
echo "  阶段2: AutoIF 环境工程数据合成"
echo "============================================"

echo "[Step 1/9] 指令增强..."
python code/1_RFT.py 2>&1 | tee logs/step1.log
echo "[Step 2/9] 验证函数生成..."
python code/2_verification_funcs_cases_generation.py 2>&1 | tee logs/step2.log
echo "[Step 3/9] 交叉验证..."
python code/3_cross_validation.py 2>&1 | tee logs/step3.log
echo "[Step 4/9] 反向翻译..."
python code/4_eval_func_backtranslator.py 2>&1 | tee logs/step4.log
echo "[Step 5/9] NLI 一致性过滤..."
python code/5_eval_func_backtranslator_filter.py 2>&1 | tee logs/step5.log
echo "[Step 6/9] 领域查询配对与响应生成..."
python code/6_concat_sharegpt_query.py 2>&1 | tee logs/step6.log
echo "[Step 7/9] 三层验证与质量评分..."
python code/7_query_verification.py 2>&1 | tee logs/step7.log
echo "[Step 8/9] 质量过滤..."
python code/8_query_score_filter.py 2>&1 | tee logs/step8.log
echo "[Step 9/9] SFT 数据构建..."
python code/9_sft_data_construction.py 2>&1 | tee logs/step9.log
echo "✅ AutoIF 数据合成完成"

echo ""
echo "[DPO-1] 响应评分..."
python code_dpo/1_dpo_rft_wash.py 2>&1 | tee logs/dpo1.log
echo "[DPO-2] 偏好对构建（保留完整环境问题+格式要求）..."
python code_dpo/2_dpo_data_query_construct.py 2>&1 | tee logs/dpo2.log
echo "[DPO-3] 环境工程硬负样本..."
python code_dpo/3_env_hard_negatives.py 2>&1 | tee logs/dpo3.log
echo "✅ DPO 数据构建完成"

echo ""
echo "--- 数据统计 ---"
for f in output/*.json output/*.jsonl data/*.jsonl sample_data/seed_instruction_环境工程.txt; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  $(basename "$f"): $count 行"
    fi
done

echo ""
echo "关闭 vLLM 服务（释放显存）..."
cleanup
VLLM_PID=""
sleep 5
echo "✅ vLLM 已关闭"

if [ "$SKIP_TRAIN" = "1" ]; then
    python scripts/build_experiment_summary.py
    echo "已跳过训练（--skip-train）"
    exit 0
fi

echo ""
echo "============================================"
echo "  阶段4: 准备训练数据 + SFT"
echo "============================================"
python scripts/prepare_llamafactory_data.py

cd LlamaFactory
llamafactory-cli train \
    --stage sft --do_train \
    --model_name_or_path "$STUDENT_PATH" \
    --dataset autoif_sft --template qwen \
    --finetuning_type lora --lora_rank 8 --lora_alpha 16 \
    --lora_target q_proj,v_proj \
    --output_dir ../models/model_c_sft --overwrite_output_dir \
    --per_device_train_batch_size "$SFT_BS" \
    --gradient_accumulation_steps "$SFT_GA" \
    --learning_rate "$SFT_LR" --num_train_epochs "$SFT_EPOCHS" \
    --logging_steps 5 --save_steps 100 --warmup_ratio 0.1 \
    --fp16 --cutoff_len 2048 --report_to none \
    2>&1 | tee ../logs/sft_train.log
cd ..
echo "✅ SFT 训练完成"

echo ""
echo "============================================"
echo "  阶段5: DPO 训练"
echo "============================================"
cd LlamaFactory
llamafactory-cli train \
    --stage dpo --do_train \
    --model_name_or_path "$STUDENT_PATH" \
    --adapter_name_or_path ../models/model_c_sft \
    --dataset autoif_dpo --template qwen \
    --finetuning_type lora --lora_rank 8 --lora_alpha 16 \
    --lora_target q_proj,v_proj \
    --output_dir ../models/model_c_dpo --overwrite_output_dir \
    --per_device_train_batch_size "$DPO_BS" \
    --gradient_accumulation_steps "$DPO_GA" \
    --learning_rate "$DPO_LR" --num_train_epochs "$DPO_EPOCHS" \
    --logging_steps 5 --save_steps 100 --warmup_ratio 0.1 \
    --fp16 --cutoff_len 2048 --pref_beta "$PREF_BETA" --report_to none \
    2>&1 | tee ../logs/dpo_train.log
cd ..
echo "✅ DPO 训练完成"

echo ""
echo "============================================"
echo "  阶段6: LoRA 权重合并"
echo "============================================"
cd LlamaFactory
llamafactory-cli export \
    --model_name_or_path "$STUDENT_PATH" \
    --adapter_name_or_path "../models/model_c_dpo" \
    --template qwen --finetuning_type lora \
    --export_dir "../models/model_merged" \
    --export_size 2 --export_legacy_format false 2>&1 | tee ../logs/merge.log
cd ..
echo "✅ LoRA 合并完成"

echo ""
echo "============================================"
echo "  阶段7: GPTQ 量化（可选，环境工程校准语料）"
echo "============================================"
python scripts/quantize_env.py 2>&1 | tee logs/quantize.log || echo "量化失败不影响合并模型，且不得预先宣称精度只损失2%~3%。"

echo ""
echo "============================================"
echo "  阶段8: 环境工程推理测试"
echo "============================================"
python scripts/smoke_infer.py 2>&1 | tee logs/inference_test.log || true

python scripts/build_experiment_summary.py | tee logs/experiment_summary.log

echo ""
echo "============================================"
echo "  全流程完成！领域: $DOMAIN  结束: $(date)"
echo "============================================"
echo "实验数字请只引用: output/experiment_summary.json"
echo "SFT LoRA: models/model_c_sft/"
echo "DPO LoRA: models/model_c_dpo/"
echo "合并模型: models/model_merged/"
du -sh models/model_c_sft/ models/model_c_dpo/ models/model_merged/ 2>/dev/null || true
