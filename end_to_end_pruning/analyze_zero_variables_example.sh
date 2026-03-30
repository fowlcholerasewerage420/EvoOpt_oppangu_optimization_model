#!/bin/bash
# LP文件零变量分析脚本（基于微调后模型）

# ========== 配置参数 ==========
# 基础模型路径（必需）
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"openPangu-Embedded-7B-V1.1\\openpangu-embedded-7b-model"}
# LoRA权重路径（可选，如果使用微调后的模型）
LORA_PATH=${LORA_PATH:-"openPangu-Embedded-7B-V1.1\finetune\train\output"}
# LP输入文件（必需）
LP_INPUT=${LP_INPUT:-"end_to_end_pruning/data/xx.lp"}
# 输出JSON文件（可选）
OUTPUT_JSON=${OUTPUT_JSON:-"openPangu-Embedded-7B-V1.1\end_to_end_pruning\results"}
# 生成的候选变量数量
TOPK=${TOPK:-1}
# 解码方法
DECODING_METHOD=${DECODING_METHOD:-"greedy"}
# 采样温度
TEMPERATURE=${TEMPERATURE:-0.7}
# Top-p采样参数
TOP_P=${TOP_P:-0.95}
# 最大生成token数
MAX_TOKENS=${MAX_TOKENS:-4096}
# 运行设备
DEVICE=${DEVICE:-"auto"}
# 使用的GPU数量
NUM_DEVICES=${NUM_DEVICES:-"auto"}
# PyTorch数据类型
TORCH_DTYPE=${TORCH_DTYPE:-"auto"}
# Python分析脚本路径
PYTHON_SCRIPT=${PYTHON_SCRIPT:-"analyze_zero_variables.py"}
# 详细输出模式
VERBOSE=${VERBOSE:-""}

# ========== 打印配置 ==========
echo "=========================================="
echo "LP文件零变量分析配置"
echo "=========================================="
echo "基础模型路径: $BASE_MODEL_PATH"
echo "LoRA权重路径: ${LORA_PATH:-'None'}"
echo "LP文件输入: $LP_INPUT"
echo "输出JSON文件: ${OUTPUT_JSON:-'None (结果将不保存)'}"
echo "Python脚本: $PYTHON_SCRIPT"
echo "TopK: $TOPK"
echo "解码方法: $DECODING_METHOD"
echo "设备: $DEVICE"
echo "设备数量: ${NUM_DEVICES:-'auto'}"
echo "数据类型: $TORCH_DTYPE"
echo "=========================================="

# ========== 检查必要参数 ==========
if [ -z "$LP_INPUT" ]; then
    echo "错误: LP_INPUT 参数未设置"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ] && [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "错误: 基础模型路径不存在: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# ========== 运行LP文件分析 ==========
echo "开始分析LP文件..."
echo "=========================================="

# 构建Python命令参数
PYTHON_ARGS=(
    "$PYTHON_SCRIPT"
    --base_model_path "$BASE_MODEL_PATH"
    --lp_input "$LP_INPUT"
    --topk $TOPK
    --decoding_method "$DECODING_METHOD"
    --temperature $TEMPERATURE
    --top_p $TOP_P
    --device "$DEVICE"
    --torch_dtype "$TORCH_DTYPE"
    --trust_remote_code
)

# 可选参数
[ -n "$LORA_PATH" ] && PYTHON_ARGS+=(--lora_path "$LORA_PATH")
[ -n "$NUM_DEVICES" ] && PYTHON_ARGS+=(--num_devices "$NUM_DEVICES")
[ -n "$MAX_TOKENS" ] && PYTHON_ARGS+=(--max_tokens $MAX_TOKENS)
[ -n "$OUTPUT_JSON" ] && PYTHON_ARGS+=(--output_json "$OUTPUT_JSON")
[ -n "$VERBOSE" ] && PYTHON_ARGS+=(--verbose)

echo "执行命令: python ${PYTHON_ARGS[*]}"
echo "=========================================="

# 执行Python脚本
python "${PYTHON_ARGS[@]}"

# 检查执行状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "LP文件分析完成！"
    echo "=========================================="
    echo ""
    if [ -n "$OUTPUT_JSON" ]; then
        echo "结果已保存到JSON文件: $OUTPUT_JSON"
        echo "文件包含所有LP文件的分析结果和可固定为0的变量列表"
    else
        echo "结果已直接输出到终端，包含可固定为0的变量列表（JSON格式）"
        echo "提示: 设置 OUTPUT_JSON 环境变量可将结果保存到文件"
    fi
    echo ""
else
    echo ""
    echo "=========================================="
    echo "LP文件分析失败！"
    echo "=========================================="
    echo "错误代码: $?"
    echo "请检查:"
    echo "1. Python脚本是否存在: $PYTHON_SCRIPT"
    echo "2. 模型路径是否正确: $BASE_MODEL_PATH"
    echo "3. 输入LP文件是否存在: $LP_INPUT"
    echo "4. Python环境是否安装必要依赖"
    exit 1
fi

echo ""
echo "=========================================="
echo "脚本执行完成"
echo "=========================================="