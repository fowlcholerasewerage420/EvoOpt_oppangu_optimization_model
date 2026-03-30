#!/bin/bash
# 微调后模型批量生成示例脚本

# ========== 配置参数 ==========
# 基础模型路径（必需）
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"openPangu-Embedded-7B-V1.1\\openpangu-embedded-7b-model"}

# LoRA权重路径（可选，如果为None则只使用基础模型）
LORA_PATH=${LORA_PATH:-""}

# 数据集路径（必需）
DATASET_NAME=${DATASET_NAME:-"path_to_your_dataset"}

# 数据集划分
DATASET_SPLIT=${DATASET_SPLIT:-"train"}

# 输出目录（必需）
SAVE_DIR=${SAVE_DIR:-"output_directory"}

# 生成参数
TOPK=${TOPK:-1}  # 每个prompt生成topk个结果
DECODING_METHOD=${DECODING_METHOD:-"greedy"}  # greedy 或 sampling
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-2048}

# 设备配置
DEVICE=${DEVICE:-"auto"}  # auto, cuda, npu, cpu
TORCH_DTYPE=${TORCH_DTYPE:-"auto"}  # auto, float16, bfloat16, float32

# ========== 打印配置 ==========
echo "=========================================="
echo "微调后模型批量生成配置"
echo "=========================================="
echo "基础模型路径: $BASE_MODEL_PATH"
echo "LoRA权重路径: ${LORA_PATH:-'None (仅使用基础模型)'}"
echo "数据集: $DATASET_NAME ($DATASET_SPLIT)"
echo "输出目录: $SAVE_DIR"
echo "TopK: $TOPK"
echo "解码方法: $DECODING_METHOD"
echo "设备: $DEVICE"
echo "数据类型: $TORCH_DTYPE"
echo "=========================================="

# ========== 检查路径 ==========
if [ ! -d "$BASE_MODEL_PATH" ] && [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "错误: 基础模型路径不存在: $BASE_MODEL_PATH"
    echo "请设置正确的BASE_MODEL_PATH环境变量"
    exit 1
fi

# ========== 运行生成 ==========
python -m generate_finetuned \
    --base_model_path "$BASE_MODEL_PATH" \
    ${LORA_PATH:+--lora_path "$LORA_PATH"} \
    --dataset_name "$DATASET_NAME" \
    --dataset_split "$DATASET_SPLIT" \
    --save_dir "$SAVE_DIR" \
    --topk $TOPK \
    --decoding_method "$DECODING_METHOD" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --trust_remote_code \
    ${VERBOSE:+--verbose}

echo ""
echo "生成完成！结果保存在: $SAVE_DIR/generated.jsonl"