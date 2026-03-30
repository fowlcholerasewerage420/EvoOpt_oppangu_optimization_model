import argparse
import json
from pathlib import Path
from typing import Set, List

def load_generated_json(generated_json_path: str) -> List[str]:
    """
    加载大模型生成的JSON文件，提取所有zero_variables
    """
    with open(generated_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 从results中提取所有zero_variables
    all_zero_variables = []
    if "results" in data:
        for result in data["results"]:
            if "zero_variables" in result:
                all_zero_variables.extend(result["zero_variables"])

    return all_zero_variables

def load_answer_json(answer_json_path: str) -> Set[str]:
    """
    加载答案文件，返回所有剪枝变量的集合
    """
    with open(answer_json_path, 'r', encoding='utf-8') as f:
        answer_list = json.load(f)

    # 转换为集合以便快速查找
    return set(answer_list)

def calculate_metrics(generated_variables: List[str], answer_variables: Set[str]) -> dict:
    """
    计算评价指标

    Args:
        generated_variables: 大模型生成的所有变量列表（可能包含重复）
        answer_variables: 答案文件中所有剪枝变量的集合

    Returns:
        包含各项指标的字典
    """
    # 去重后的生成变量
    generated_unique = set(generated_variables)

    # 计算正确的变量（在答案文件中的变量）
    correct_variables = generated_unique & answer_variables

    # 计算指标
    total_answer_variables = len(answer_variables)
    total_generated_unique = len(generated_unique)
    total_correct = len(correct_variables)

    # 精确率：正确变量占生成变量的比例
    precision = total_correct / total_generated_unique if total_generated_unique > 0 else 0.0

    # 召回率：正确变量占答案变量的比例
    recall = total_correct / total_answer_variables if total_answer_variables > 0 else 0.0

    # F1分数
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_answer_variables": total_answer_variables,
        "total_generated_variables": len(generated_variables),
        "total_generated_unique": total_generated_unique,
        "total_correct": total_correct,
        "total_incorrect": total_generated_unique - total_correct,
        "precision": precision,  # 精确率：正确变量占生成变量的比例
        "recall": recall,  # 召回率：正确变量占答案变量的比例
        "f1_score": f1_score  # F1分数
    }

def main(args):
    print("=" * 80)
    print("固定为0的变量准确性评价")
    print("=" * 80)
    print(f"生成的JSON文件: {args.generated_json}")
    print(f"答案文件: {args.answer_json}")
    print("=" * 80)

    # 加载数据
    print("\n正在加载数据...")
    generated_variables = load_generated_json(args.generated_json)
    answer_variables = load_answer_json(args.answer_json)

    print(f"已加载 {len(generated_variables)} 个生成的变量（含重复）")
    print(f"已加载 {len(answer_variables)} 个答案变量")

    # 计算指标
    print("\n正在计算评价指标...")
    metrics = calculate_metrics(generated_variables, answer_variables)

    # 打印结果
    print("\n" + "=" * 80)
    print("评价结果")
    print("=" * 80)
    print(f"答案文件中的总变量数: {metrics['total_answer_variables']}")
    print(f"生成的变量总数（含重复）: {metrics['total_generated_variables']}")
    print(f"生成的变量总数（去重后）: {metrics['total_generated_unique']}")
    print(f"正确的变量数: {metrics['total_correct']}")
    print(f"错误的变量数: {metrics['total_incorrect']}")
    print("-" * 80)
    print(f"精确率 (Precision): {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  说明: 正确变量占生成变量的比例")
    print(f"召回率 (Recall): {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  说明: 正确变量占答案变量的比例")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print("=" * 80)

    # 保存结果到JSON文件
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "input_files": {
                "generated_json": args.generated_json,
                "answer_json": args.answer_json
            },
            "metrics": metrics,
            "details": {
                "generated_variables_count": len(generated_variables),
                "generated_unique_variables_count": metrics['total_generated_unique'],
                "answer_variables_count": metrics['total_answer_variables'],
                "correct_variables_count": metrics['total_correct'],
                "incorrect_variables_count": metrics['total_incorrect']
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_path}")

    print()

def parse_args():
    parser = argparse.ArgumentParser(
        description="评价大模型生成的固定为0的变量的准确性"
    )

    parser.add_argument(
        "--generated_json",
        type=str,
        required=True,
        help="大模型生成的JSON文件路径（包含zero_variables字段）"
    )

    parser.add_argument(
        "--answer_json",
        type=str,
        required=True,
        help="答案文件路径（包含所有剪枝变量的JSON数组）"
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出JSON文件路径（保存评价结果，可选）"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)