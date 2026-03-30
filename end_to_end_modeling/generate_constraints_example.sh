#!/bin/bash
# LP文件约束修改脚本（基于微调后模型）

# ========== 配置参数 ==========
# 基础模型路径（必需）
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"openPangu-Embedded-7B-V1.1\\openpangu-embedded-7b-model"}
# LoRA权重路径（可选）
LORA_PATH=${LORA_PATH:-"openPangu-Embedded-7B-V1.1\finetune\train\output"}
# LP输入文件路径（必需）
LP_INPUT=${LP_INPUT:-"end_to_end_modeling/lp"}
# 修改需求描述（必需）
REQUIREMENTS=${REQUIREMENTS:-"The existing MILP model for supply chain scheduling has no human resource-related constraints. I need to supplement the model with human resource constraints. Specific requirements:

1. **Differential labor requirements**: Different processes/equipment require different amounts of labor for operation.
2. **Total labor limitation**: The total available labor in each time period (day) is limited, and this limit can be time-varying.
3. **Total labor constraint**: At any given time, the sum of labor consumed by all processes cannot exceed the available labor for that time period.
4. **Variable association**: Can introduce 'whether in operation' binary variables to associate with labor consumption, i.e., labor is only consumed when a process is operating.
5. **Modeling flexibility**: Allow modeling of different skill level labor requirements (optional, if needed).

Please provide based on the existing model framework:
- New set definitions (e.g., human resource type set H)
- New parameter definitions (e.g., labor required for process p on equipment m: r_{p,m}, available labor in time period t: R_t)
- New variable definitions (e.g., labor consumption variables)
- Corresponding linear constraint formulas
- Detailed explanation of modeling logic

The existing model includes: time index t, process p, equipment m, production decision variables, etc."}
# 输出目录（必需）
SAVE_DIR=${SAVE_DIR:-"output_directory"}
# 生成的候选数量
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
# Python脚本路径
PYTHON_SCRIPT=${PYTHON_SCRIPT:-"generate_constraints.py"}
# 详细输出模式
VERBOSE=${VERBOSE:-""}
# 覆盖模式
OVERWRITE=${OVERWRITE:-""}

# ========== 打印配置 ==========
echo "=========================================="
echo "LP文件约束修改配置"
echo "=========================================="
echo "基础模型路径: $BASE_MODEL_PATH"
echo "LoRA权重路径: ${LORA_PATH:-'None'}"
echo "LP文件输入: $LP_INPUT"
echo "输出目录: $SAVE_DIR"
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

if [ -z "$REQUIREMENTS" ]; then
    echo "错误: REQUIREMENTS 参数未设置"
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

# ========== 创建输出目录 ==========
mkdir -p "$SAVE_DIR"
echo "输出目录已创建: $SAVE_DIR"

# ========== 运行LP文件修改 ==========
echo "开始处理LP文件..."
echo "=========================================="

# 构建Python命令参数
PYTHON_ARGS=(
    "$PYTHON_SCRIPT"
    --base_model_path "$BASE_MODEL_PATH"
    --lp_input "$LP_INPUT"
    --requirements "$REQUIREMENTS"
    --save_dir "$SAVE_DIR"
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
[ -n "$VERBOSE" ] && PYTHON_ARGS+=(--verbose)
[ -n "$OVERWRITE" ] && PYTHON_ARGS+=(--overwrite)

echo "执行命令: python ${PYTHON_ARGS[*]}"
echo "=========================================="

# 执行Python脚本
python "${PYTHON_ARGS[@]}"

# 检查执行状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "LP文件修改完成！"
    echo "=========================================="
    echo "输出目录: $SAVE_DIR"
    echo ""

    # 检查结果文件
    RESULTS_JSON="$SAVE_DIR/modified_lp_results.json"

    if [ -f "$RESULTS_JSON" ]; then
        echo "主要结果文件:"
        echo "  $RESULTS_JSON"
        echo ""

        # 使用Python解析JSON结果
        echo "处理摘要:"
        python3 -c "
import json
import os
from pathlib import Path

# 加载结果文件
with open('$RESULTS_JSON', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 基本统计
total = data['metadata']['total_files']
successful = sum(1 for r in data['results'] if r['status'] == 'success')
failed = total - successful
saved_lp_files = sum(1 for r in data['results'] if r.get('lp_output_path') and r['status'] == 'success')

print(f'总文件数: {total}')
print(f'成功: {successful}')
print(f'失败: {failed}')
print(f'生成LP文件: {saved_lp_files}')
print(f'成功率: {successful/total*100:.1f}%')

# 如果有成功的结果，显示详细信息
if successful > 0:
    # 找到第一个成功的结果
    first_success = next((r for r in data['results'] if r['status'] == 'success'), None)
    if first_success:
        content = first_success.get('modified_content', '')
        print(f'\n第一个成功结果:')
        print(f'  原始文件: {Path(first_success['file_path']).name}')
        print(f'  修改内容长度: {len(content)} 字符')

# 显示生成的.lp文件列表
lp_files = [r.get('lp_output_path') for r in data['results'] if r.get('lp_output_path')]
if lp_files:
    print(f'\n生成的.lp文件列表:')
    for lp_file in lp_files:
        if os.path.exists(lp_file):
            filename = Path(lp_file).name
            size = os.path.getsize(lp_file)
            print(f'  - {filename} ({size:,} bytes)')
        else:
            print(f'  - {Path(lp_file).name} (文件不存在)')
"
        echo ""

        # 显示.lp文件内容（第一个文件的预览）
        echo "第一个生成的.lp文件内容预览:"
        echo "------------------------------------------"
        FIRST_LP=$(find "$SAVE_DIR" -name "*_modified.lp" -type f | head -1)
        if [ -n "$FIRST_LP" ] && [ -f "$FIRST_LP" ]; then
            echo "文件: $(basename "$FIRST_LP")"
            echo ""
            # 显示前20行
            head -20 "$FIRST_LP"
            echo "..."
            echo "------------------------------------------"
            echo "完整文件: $FIRST_LP"
        else
            # 如果没有找到_modified.lp文件，查找.lp扩展名的文件
            FIRST_LP=$(find "$SAVE_DIR" -name "*.lp" -type f | grep -v "original" | head -1)
            if [ -n "$FIRST_LP" ] && [ -f "$FIRST_LP" ]; then
                echo "文件: $(basename "$FIRST_LP")"
                echo ""
                # 显示前20行
                head -20 "$FIRST_LP"
                echo "..."
                echo "------------------------------------------"
                echo "完整文件: $FIRST_LP"
            else
                echo "未找到.lp文件"
            fi
        fi
    else
        echo "警告: 结果文件不存在: $RESULTS_JSON"
        echo "检查Python脚本是否成功执行"
    fi

    # 显示目录中的文件列表
    echo ""
    echo "输出目录中的文件:"
    echo "------------------------------------------"
    ls -la "$SAVE_DIR" | while read line; do
        echo "  $line"
    done

    # 如果有.lp文件，提供使用建议
    LP_COUNT=$(find "$SAVE_DIR" -name "*.lp" -type f | wc -l)
    if [ $LP_COUNT -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "使用建议:"
        echo "=========================================="
        echo "1. 生成的.lp文件可以直接用于求解器:"
        echo "   lp_solve /path/to/generated_file.lp"
        echo ""
        echo "2. 验证模型完整性:"
        echo "   python -c \"with open('$FIRST_LP', 'r') as f: "
        echo "        content = f.read(); "
        echo "        print('目标函数:', 'Minimize' in content or 'Maximize' in content); "
        echo "        print('约束数量:', content.count('<= ') + content.count('= ') + content.count('>= '))\""
        echo ""
        echo "3. 批量处理多个LP文件:"
        echo "   for lp_file in $SAVE_DIR/*.lp; do"
        echo "     echo \"处理: \$lp_file\""
        echo "     lp_solve \"\$lp_file\" > \"\${lp_file%.lp}.result\""
        echo "   done"
    fi
else
    echo ""
    echo "=========================================="
    echo "LP文件修改失败！"
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