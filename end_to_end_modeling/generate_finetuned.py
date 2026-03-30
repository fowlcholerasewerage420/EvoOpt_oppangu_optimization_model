import argparse
import json
import os
import sys
import datasets
import torch

# NPU support
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

def load_model_and_tokenizer(base_model_path, lora_path=None, device="auto", torch_dtype="auto", trust_remote_code=True):
    """
    加载基础模型和LoRA权重
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径（如果为None，则只加载基础模型）
        device: 设备类型 ("auto", "cuda", "npu", "cpu")
        torch_dtype: 数据类型 ("auto", "float16", "bfloat16", "float32")
        trust_remote_code: 是否信任远程代码
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model from: {base_model_path}")
    
    # 确定设备
    if device == "auto":
        if NPU_AVAILABLE and torch_npu.npu.is_available():
            device = "npu"
            torch.npu.set_device(0)
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # 确定数据类型
    if torch_dtype == "auto":
        if device == "npu":
            torch_dtype = torch.bfloat16
        elif device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        use_fast=False
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # 加载基础模型
    print("Loading base model...")
    if device == "npu":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=None,
        )
        model = model.to("npu:0")
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
        )
    
    print("Base model loaded successfully.")
    
    # 加载LoRA权重
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        try:
            if device == "npu":
                # NPU环境的特殊处理
                model = model.to("npu:0")
                torch.npu.set_device(0)
                
                # 临时将模型移动到CPU加载LoRA权重
                print("Moving model to CPU for LoRA weight loading...")
                model_cpu = model.to("cpu")
                
                # 临时禁用CUDA检测
                original_cuda_available = None
                if hasattr(torch.cuda, 'is_available'):
                    original_cuda_available = torch.cuda.is_available
                    torch.cuda.is_available = lambda: False
                
                try:
                    print("Loading LoRA weights on CPU...")
                    model_cpu = PeftModel.from_pretrained(
                        model_cpu, 
                        lora_path,
                        device_map=None,
                    )
                    print("LoRA weights loaded on CPU successfully.")
                finally:
                    if original_cuda_available is not None:
                        torch.cuda.is_available = original_cuda_available
                
                # 将模型移回NPU
                print("Moving model with LoRA weights to NPU...")
                model = model_cpu.to("npu:0")
                print("Model moved to NPU successfully.")
            else:
                # CUDA或CPU环境
                model = PeftModel.from_pretrained(model, lora_path)
                print("LoRA weights loaded successfully.")
            
            # 如果指定了merge_and_unload，合并LoRA权重到基础模型
            # 注意：这里我们不自动合并，保持LoRA权重分离以便灵活使用
        except Exception as e:
            import traceback
            print(f"Warning: Failed to load LoRA weights: {e}")
            print(f"Error details: {traceback.format_exc()}")
            print("Continuing with base model only.")
    else:
        if lora_path is not None:
            print(f"Warning: LoRA path does not exist: {lora_path}")
            print("Continuing with base model only.")
    
    # 最终确保模型在正确的设备上
    if device == "npu":
        model = model.to("npu:0")
        torch.npu.set_device(0)
    
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.95, 
                  do_sample=False, stop_tokens=None, device="auto", topk=1):
    """
    生成文本（支持topk采样）
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 温度参数（用于采样）
        top_p: top_p参数（用于采样）
        do_sample: 是否使用采样
        stop_tokens: 停止token列表
        device: 设备类型
        topk: 生成topk个结果
    
    Returns:
        生成的文本列表（长度为topk）
    """
    # 确定设备
    if device == "auto":
        if hasattr(model, 'device'):
            device = str(model.device)
        elif NPU_AVAILABLE and torch_npu.npu.is_available():
            device = "npu:0"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    
    # Tokenize输入
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # 移动到设备
    if device.startswith("npu"):
        npu_device = "npu:0" if ":" not in device else device
        inputs = {k: v.to(npu_device) for k, v in inputs.items()}
    elif device.startswith("cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成参数
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": topk,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if do_sample:
        generation_config["temperature"] = temperature
        generation_config["top_p"] = top_p
    
    # 生成
    outputs_list = []
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    
    # 确保outputs是2D张量 [batch_size, seq_len]
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    
    # 解码所有生成结果
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        # 移除输入部分，只返回生成的部分
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        outputs_list.append(generated_text)
    
    return outputs_list

def main(args):
    assert args.dataset_name is not None
    assert args.dataset_split is not None
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    assert os.path.exists(args.base_model_path), "Base model path must exist!"
    assert args.save_dir is not None

    os.makedirs(args.save_dir, exist_ok=True)

    # 确定设备
    device = args.device
    if device == "auto":
        if NPU_AVAILABLE and torch_npu.npu.is_available():
            device = "npu"
            torch.npu.set_device(0)
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"=" * 80)
    print(f"Generation Configuration:")
    print(f"  Base Model: {args.base_model_path}")
    print(f"  LoRA Path: {args.lora_path if args.lora_path else 'None (base model only)'}")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_name} ({args.dataset_split})")
    print(f"  Decoding Method: {args.decoding_method}")
    print(f"  TopK: {args.topk}")
    print(f"=" * 80)

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        device=device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code
    )

    # Load data
    sample = []
    print(f"\nLoading dataset from `{args.dataset_name}`...")
    ds = datasets.load_dataset(args.dataset_name)
    ds = ds[args.dataset_split]
    for example in ds:
        assert "en_question" in example
        prompt = TEMPLATE_q2mc_en.replace("{Question}", example["en_question"].strip()).strip()
        example_t = {k: v for k, v in example.items() if k not in ["prompt"]}
        example_t["prompt"] = prompt
        sample.append(example_t)
    print(f"Dataset loaded. Sample size: {len(ds)}")

    # 设置生成参数
    do_sample = (args.decoding_method == "sampling")
    temperature = args.temperature if do_sample else 0.0
    top_p = args.top_p if do_sample else 1.0
    
    stop_tokens = args.stop_tokens.split(",") if args.stop_tokens else None

    # Generate
    save_file = os.path.join(args.save_dir, "generated.jsonl")
    fw = open(save_file, "w", encoding='utf-8')
    num_total = 0
    num_skip_for_dup = 0
    
    print(f"\nGenerating responses...")
    print(f"Generation config: do_sample={do_sample}, temperature={temperature}, top_p={top_p}")
    
    for example in tqdm(sample, desc="Generating"):
        prompt = example["prompt"]
        try:
            outputs = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens if args.max_tokens else 2048,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                stop_tokens=stop_tokens,
                device=device,
                topk=args.topk
            )
            
            outputs_t = []
            touched_output = set()
            for output in outputs:
                num_total += 1
                if output not in touched_output:
                    outputs_t.append(output)
                    touched_output.add(output)
                else:
                    num_skip_for_dup += 1

            for output in outputs_t:
                example_t = {k: v for k, v in example.items()}
                example_t["q2mc_en_prompt"] = prompt
                example_t["en_math_model_coptpy_code"] = output
                if args.verbose:
                    print("-" * 20 + "prompt" + "-" * 20)
                    print(prompt)
                    print("-" * 20 + "completion" + "-" * 20)
                    print(output)
                    print("-" * 80)

                dump = json.dumps(example_t, ensure_ascii=False)
                fw.write(dump + "\n")
        except Exception as e:
            print(f"Error generating for example: {e}")
            import traceback
            traceback.print_exc()
            # 保存错误信息
            example_t = {k: v for k, v in example.items()}
            example_t["q2mc_en_prompt"] = prompt
            example_t["en_math_model_coptpy_code"] = f"ERROR: {str(e)}"
            dump = json.dumps(example_t, ensure_ascii=False)
            fw.write(dump + "\n")
    
    fw.close()
    print(f"\nGeneration completed!")
    print(f"Total generations: {num_total}; Skipped duplicates: {num_skip_for_dup}")
    print(f"Results saved to: {save_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="微调后模型批量生成脚本")
    
    # 模型路径
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA权重路径（如果为None，则只使用基础模型）")
    
    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="数据集名称（HuggingFace datasets）")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="数据集分割（如 train, test, validation）")
    
    # 输出配置
    parser.add_argument("--save_dir", type=str, default=None,
                        help="保存目录")
    
    # 生成参数
    parser.add_argument("--topk", type=int, default=1,
                        help="每个prompt生成topk个结果")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["greedy", "sampling"],
                        help="解码方法：greedy（贪婪）或sampling（采样）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数（用于采样）")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="top_p参数（用于采样）")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="最大生成token数（默认2048）")
    parser.add_argument("--stop_tokens", type=str, default="</s>",
                        help="停止token列表（逗号分隔）")
    
    # 设备配置
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "npu", "cpu"],
                        help="设备类型（auto自动检测）")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="数据类型（auto自动选择）")
    
    # 其他
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="是否信任远程代码")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细信息")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

