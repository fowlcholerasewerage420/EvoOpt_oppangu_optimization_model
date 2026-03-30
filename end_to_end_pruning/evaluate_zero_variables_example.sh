#!/bin/bash
# 固定为0的变量准确性评价脚本

# ========== 配置参数 ==========
# 大模型生成的JSON文件路径（必需）
GENERATED_JSON=${GENERATED_JSON:-"end_to_end_pruning/results/xx.json"}
# 答案文件路径（必需）
ANSWER_JSON=${ANSWER_JSON:-"end_to_end_pruning/answer/xx.json"}
# 输出JSON文件路径（可选）
OUTPUT_JSON=${OUTPUT_JSON:-"end_to_end_pruning/results"}
# Python评价脚本路径
PYTHON_SCRIPT=${PYTHON_SCRIPT:-"evaluate_zero_variables.py"}

# ========== 打印配置 ==========
echo "=========================================="
echo "固定为0的变量准确性评价配置"
echo "=========================================="
echo "生成的JSON文件: ${GENERATED_JSON:-'未设置'}"
echo "答案文件: ${ANSWER_JSON:-'未设置'}"
echo "输出JSON文件: ${OUTPUT_JSON:-'None (结果将不保存)'}"
echo "Python脚本: $PYTHON_SCRIPT"
echo "=========================================="

# ========== 检查必要参数 ==========
if [ -z "$GENERATED_JSON" ]; then
    echo "错误: GENERATED_JSON 参数未设置"
    echo "用法: GENERATED_JSON=<生成的JSON文件路径> ANSWER_JSON=<答案文件路径> [OUTPUT_JSON=<输出JSON文件路径>] $0"
    exit 1
fi

if [ -z "$ANSWER_JSON" ]; then
    echo "错误: ANSWER_JSON 参数未设置"
    echo "用法: GENERATED_JSON=<生成的JSON文件路径> ANSWER_JSON=<答案文件路径> [OUTPUT_JSON=<输出JSON文件路径>] $0"
    exit 1
fi

if [ ! -f "$GENERATED_JSON" ]; then
    echo "错误: 生成的JSON文件不存在: $GENERATED_JSON"
    exit 1
fi

if [ ! -f "$ANSWER_JSON" ]; then
    echo "错误: 答案文件不存在: $ANSWER_JSON"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# ========== 运行评价 ==========
echo "开始评价..."
echo "=========================================="

# 构建Python命令参数
PYTHON_ARGS=(
    "$PYTHON_SCRIPT"
    --generated_json "$GENERATED_JSON"
    --answer_json "$ANSWER_JSON"
)

# 可选参数
[ -n "$OUTPUT_JSON" ] && PYTHON_ARGS+=(--output_json "$OUTPUT_JSON")

echo "执行命令: python ${PYTHON_ARGS[*]}"
echo "=========================================="

# 执行Python脚本
python "${PYTHON_ARGS[@]}"

# 检查执行状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "评价完成！"
    echo "=========================================="
    echo ""
    if [ -n "$OUTPUT_JSON" ]; then
        echo "结果已保存到JSON文件: $OUTPUT_JSON"
    fi
    echo ""
else
    echo ""
    echo "=========================================="
    echo "评价失败！"
    echo "=========================================="
    echo "错误代码: $?"
    echo "请检查:"
    echo "1. Python脚本是否存在: $PYTHON_SCRIPT"
    echo "2. 生成的JSON文件是否存在: $GENERATED_JSON"
    echo "3. 答案文件是否存在: $ANSWER_JSON"
    echo "4. Python环境是否安装必要依赖"
    exit 1
fi

echo ""
echo "=========================================="
echo "脚本执行完成"
echo "=========================================="