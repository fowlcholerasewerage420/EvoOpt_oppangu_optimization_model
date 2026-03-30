import argparse
import json
import os
import sys
import torch
import re
from pathlib import Path

# NPU support
try:
    import torch_npu

    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

TEMPLATE_ANALYSIS = """Below is a linear programming model provided in .lp format. Your task is to analyze the model structure, including the objective function, constraints, variable bounds, and logical relationships, and determine which decision variables can be safely fixed to 0 without affecting the feasibility or optimality of the model.

You should base your reasoning on:

- Variable bounds and sign restrictions

- Whether the variable appears in any constraint or objective

- Dominance or redundancy implied by constraints

- Logical infeasibility (e.g., variables that can never take positive values under any feasible solution)

- Structural properties of the model (e.g., mutually exclusive paths, inactive arcs, unreachable nodes)

# Input (.lp model):

{lp_content}

# Output requirements:

- Do NOT provide explanations, derivations, or intermediate reasoning

- Do NOT output code

- Do NOT include any text other than the final result

- The output must be a JSON-style list of strings

- Each string must be exactly in the form: "variable_name=0"

- Only include variables that can be fixed to 0 with certainty

# Example output format:

[

  "z(30_119_173729_1)=0",

  "z(31_119_173729_1)=0",

  "z(32_119_173729_1)=0"

]

"""

def load_model_and_tokenizer(base_model_path, lora_path=None, device="auto", torch_dtype="auto",
                             trust_remote_code=True, num_devices=None, verbose=False):
    """
    加载基础模型和LoRA权重
    支持多卡模型并行（model parallelism）
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

    # 检测可用设备数量
    if num_devices is None or num_devices == "auto":
        if device == "npu" and NPU_AVAILABLE:
            num_devices = torch_npu.npu.device_count()
        elif device == "cuda":
            num_devices = torch.cuda.device_count()
        else:
            num_devices = 1
    else:
        num_devices = int(num_devices)

    # 限制设备数量不超过实际可用数量
    if device == "npu" and NPU_AVAILABLE:
        num_devices = min(num_devices, torch_npu.npu.device_count())
    elif device == "cuda":
        num_devices = min(num_devices, torch.cuda.device_count())

    print(f"Using device: {device}, dtype: {torch_dtype}, num_devices: {num_devices}")

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
        if num_devices > 1:
            # 多NPU：尝试使用device_map自动分配
            print(f"Using {num_devices} NPUs for model parallelism")
            try:
                # 尝试使用accelerate库的device_map（如果支持NPU）
                from accelerate import infer_auto_device_map
                from transformers import AutoConfig

                # 获取模型配置
                config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)

                # 创建多NPU的device_map
                max_memory = {f"cuda:{i}": "20GiB" for i in range(num_devices)}
                max_memory["cpu"] = "100GiB"

                # 先加载模型到CPU以计算device_map
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )

                # 计算device_map（基于CUDA，然后转换为NPU）
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=getattr(config, "_no_split_modules", None) or [],
                    dtype=torch_dtype,
                )

                # 将device_map中的"cuda"替换为"npu"
                npu_device_map = {}
                for key, value in device_map.items():
                    if isinstance(value, str) and value.startswith("cuda"):
                        npu_id = int(value.split(":")[-1]) % num_devices
                        npu_device_map[key] = f"npu:{npu_id}"
                    elif isinstance(value, int):
                        npu_id = value % num_devices
                        npu_device_map[key] = npu_id
                    else:
                        npu_device_map[key] = value

                print(f"Device map created with {len(npu_device_map)} layers")
                if verbose:
                    print(f"Device map sample: {list(npu_device_map.items())[:5]}...")

                # 重新加载模型使用device_map
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    device_map=npu_device_map,
                )
                print("Model loaded with multi-NPU device_map successfully")
            except Exception as e:
                print(f"Warning: Multi-NPU device_map failed: {e}")
                print("Note: NPU multi-device support may require manual device assignment")
                print("Falling back to single NPU...")
                # 回退到单卡
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    device_map=None,
                )
                model = model.to("npu:0")
                num_devices = 1  # 更新实际使用的设备数
        else:
            # 单NPU
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=None,
            )
            model = model.to("npu:0")
    elif device == "cuda":
        if num_devices > 1:
            # 多CUDA：使用device_map="auto"自动分配
            print(f"Using {num_devices} CUDA devices for model parallelism")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map="auto",
                max_memory={i: "20GiB" for i in range(num_devices)},
            )
        else:
            # 单CUDA
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

    # 加载LoRA权重（修复版本）
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        try:
            if device == "npu":
                # 对于NPU设备，先完全在CPU上加载LoRA权重
                print("Loading LoRA weights on CPU (for NPU compatibility)...")

                # 先将模型移到CPU
                model_cpu = model.to("cpu")

                # 禁用CUDA检测
                import transformers
                original_cuda_flag = None
                if hasattr(torch.cuda, '_is_initialized'):
                    original_cuda_flag = torch.cuda._is_initialized
                    torch.cuda._is_initialized = lambda: False

                # 临时禁用CUDA可用性检测
                original_cuda_available = torch.cuda.is_available
                torch.cuda.is_available = lambda: False

                try:
                    # 在CPU上加载LoRA权重
                    model_cpu = PeftModel.from_pretrained(
                        model_cpu,
                        lora_path,
                        device_map=None,
                        torch_dtype=torch_dtype
                    )
                    print("LoRA weights loaded on CPU successfully.")
                except Exception as e:
                    print(f"Error loading LoRA on CPU: {e}")
                    print("Trying alternative method...")

                    # 尝试另一种方法：直接加载权重文件
                    try:
                        from peft import PeftConfig
                        config = PeftConfig.from_pretrained(lora_path)

                        # 创建PeftModel但不加载权重
                        model_cpu = PeftModel(model_cpu, config)

                        # 手动加载权重
                        import safetensors.torch as safetensors
                        weights_path = os.path.join(lora_path, "adapter_model.safetensors")
                        if os.path.exists(weights_path):
                            weights = safetensors.load_file(weights_path)
                            model_cpu.load_state_dict(weights, strict=False)
                            print("LoRA weights loaded via safetensors.")
                        else:
                            weights_path = os.path.join(lora_path, "adapter_model.bin")
                            if os.path.exists(weights_path):
                                weights = torch.load(weights_path, map_location="cpu")
                                model_cpu.load_state_dict(weights, strict=False)
                                print("LoRA weights loaded via pytorch.")
                    except Exception as e2:
                        print(f"Alternative method also failed: {e2}")
                        raise
                finally:
                    # 恢复CUDA检测
                    torch.cuda.is_available = original_cuda_available
                    if original_cuda_flag is not None:
                        torch.cuda._is_initialized = original_cuda_flag

                # 将模型移回NPU
                print("Moving model with LoRA weights to NPU...")
                if num_devices > 1:
                    # 多NPU：需要重新应用device_map
                    print(f"Warning: Multi-NPU LoRA loading may need manual device assignment")
                    model = model_cpu.to("npu:0")
                else:
                    model = model_cpu.to("npu:0")
                print("Model moved to NPU successfully.")

            else:
                # 对于CUDA或CPU环境，正常加载
                model = PeftModel.from_pretrained(model, lora_path)
                print("LoRA weights loaded successfully.")

        except Exception as e:
            import traceback
            print(f"Warning: Failed to load LoRA weights: {str(e)[:200]}")
            if verbose:
                print(f"Full error: {traceback.format_exc()}")
            print("Continuing with base model only.")
    else:
        if lora_path is not None:
            print(f"Warning: LoRA path does not exist: {lora_path}")
            print("Continuing with base model only.")

    # 最终确保模型在正确的设备上（仅单卡时需要）
    # 多卡模型已经通过device_map分配，不需要手动移动
    if device == "npu" and num_devices == 1:
        model = model.to("npu:0")
        torch.npu.set_device(0)

    model.eval()

    # 返回模型和设备信息
    return model, tokenizer, device, num_devices


def generate_text(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.95,
                  do_sample=False, device="auto", topk=1, num_devices=1):
    """
    生成文本
    支持多卡模型的输入设备自动检测
    """
    # 确定输入设备：对于多卡模型，找到第一个设备
    input_device = None

    if device == "auto":
        # 检查模型是否有device_map（多卡模型）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # 多卡模型：找到第一个设备
            first_device = None
            for module_device in model.hf_device_map.values():
                if isinstance(module_device, str):
                    first_device = module_device
                    break
                elif isinstance(module_device, int):
                    # 根据设备类型推断
                    if NPU_AVAILABLE and torch_npu.npu.is_available():
                        first_device = f"npu:{module_device}"
                    elif torch.cuda.is_available():
                        first_device = f"cuda:{module_device}"
                    else:
                        first_device = "cpu"
                    break
            if first_device:
                input_device = first_device
            else:
                # 回退：检查模型的第一个参数所在的设备
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    input_device = str(first_param.device)
        elif hasattr(model, 'device'):
            input_device = str(model.device)

        if input_device is None:
            # 最终回退：根据可用设备选择
            if NPU_AVAILABLE and torch_npu.npu.is_available():
                input_device = "npu:0"
            elif torch.cuda.is_available():
                input_device = "cuda:0"
            else:
                input_device = "cpu"
    else:
        input_device = device

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # 将输入移动到正确的设备
    if input_device.startswith("npu"):
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
    elif input_device.startswith("cuda"):
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
    # CPU设备不需要移动

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

    outputs_list = []
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)

    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        outputs_list.append(generated_text)

    return outputs_list

def extract_json_from_output(output_text):
    """
    从模型输出中提取JSON格式的变量列表
    尝试提取类似 ["var1=0", "var2=0"] 的格式
    """
    # 尝试直接解析JSON
    try:
        # 查找JSON数组
        json_pattern = r'\[[\s\S]*?\]'
        match = re.search(json_pattern, output_text)
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
    except:
        pass

    # 如果JSON解析失败，尝试提取所有 "variable_name=0" 格式的行
    pattern = r'["\']?([^"\'\s]+)=0["\']?'
    matches = re.findall(pattern, output_text)
    if matches:
        # 确保格式为 "variable_name=0"
        result = [f"{match}=0" for match in matches]
        return result

    return []

def process_lp_file(lp_file_path, model, tokenizer, device="auto",
                    max_new_tokens=2048, do_sample=False, temperature=0.7, top_p=0.95, num_devices=1):
    """
    处理单个LP文件，返回固定为0的变量列表
    """
    try:
        with open(lp_file_path, 'r', encoding='utf-8') as f:
            lp_content = f.read().strip()
    except Exception as e:
        print(f"Error reading LP file {lp_file_path}: {e}")
        return None

    prompt = TEMPLATE_ANALYSIS.format(lp_content=lp_content)

    try:
        outputs = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            topk=1,
            num_devices=num_devices
        )

        if outputs:
            output_text = outputs[0]
            # 提取JSON格式的变量列表
            zero_variables = extract_json_from_output(output_text)
            return {
                "file_path": lp_file_path,
                "raw_output": output_text,
                "zero_variables": zero_variables,
                "status": "success"
            }
        else:
            return {
                "file_path": lp_file_path,
                "raw_output": "",
                "zero_variables": [],
                "status": "failed",
                "error": "No output generated"
            }

    except Exception as e:
        print(f"Error analyzing LP file {lp_file_path}: {e}")
        return {
            "file_path": lp_file_path,
            "raw_output": "",
            "zero_variables": [],
            "status": "failed",
            "error": str(e)
        }

def main(args):
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
    print(f"LP File Zero Variable Analysis Configuration:")
    print(f"  Base Model: {args.base_model_path}")
    print(f"  LoRA Path: {args.lora_path if args.lora_path else 'None (base model only)'}")
    print(f"  Device: {device}")
    print(f"  Number of Devices: {args.num_devices if args.num_devices else 'auto'}")
    print(f"  LP Input: {args.lp_input}")
    print(f"  Decoding Method: {args.decoding_method}")
    print(f"  TopK: {args.topk}")
    print(f"  Output JSON: {args.output_json if args.output_json else 'None (results will not be saved)'}")
    print(f"=" * 80)

    model, tokenizer, device_type, num_devices = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        device=device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        num_devices=args.num_devices,
        verbose=args.verbose
    )

    do_sample = (args.decoding_method == "sampling")
    temperature = args.temperature if do_sample else 0.0
    top_p = args.top_p if do_sample else 1.0

    lp_files = []

    if os.path.isfile(args.lp_input) and args.lp_input.endswith('.lp'):
        lp_files = [args.lp_input]
    elif os.path.isdir(args.lp_input):
        lp_files = [str(f) for f in Path(args.lp_input).glob('**/*.lp')]
    else:
        try:
            with open(args.lp_input, 'r', encoding='utf-8') as f:
                lp_files = [line.strip() for line in f if line.strip().endswith('.lp')]
        except:
            print(f"Error: {args.lp_input} is not a valid file or directory")
            return

    if not lp_files:
        print(f"No LP files found in {args.lp_input}")
        return

    print(f"\nFound {len(lp_files)} LP file(s) to process")

    results = []

    for lp_file_path in tqdm(lp_files, desc="Processing LP files"):
        print(f"\nProcessing: {lp_file_path}")

        result = process_lp_file(
            lp_file_path=lp_file_path,
            model=model,
            tokenizer=tokenizer,
            device=device_type,
            max_new_tokens=args.max_tokens if args.max_tokens else 2048,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_devices=num_devices
        )

        if result:
            results.append(result)

            # 直接输出结果
            print(f"\n{'=' * 80}")
            print(f"Results for: {Path(lp_file_path).name}")
            print(f"{'=' * 80}")

            if result["status"] == "success":
                zero_vars = result["zero_variables"]
                print(f"\nFound {len(zero_vars)} variables that can be fixed to 0:")
                print("\n" + json.dumps(zero_vars, indent=2, ensure_ascii=False))

                if args.verbose:
                    print(f"\nRaw model output:")
                    print(
                        result["raw_output"][:500] + "..." if len(result["raw_output"]) > 500 else result["raw_output"])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                if args.verbose and result.get("raw_output"):
                    print(f"\nRaw model output:")
                    print(
                        result["raw_output"][:500] + "..." if len(result["raw_output"]) > 500 else result["raw_output"])
            print(f"{'=' * 80}\n")

    # 打印总结
    print(f"\n{'=' * 80}")
    print(f"Processing Summary:")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(lp_files)}")
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "failed"])
    total_zero_vars = sum(len(r["zero_variables"]) for r in results if r["status"] == "success")

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total zero variables found: {total_zero_vars}")
    print(f"{'=' * 80}\n")

    # 保存结果到JSON文件
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 构建输出数据结构
        output_data = {
            "summary": {
                "total_files": len(lp_files),
                "successful": successful,
                "failed": failed,
                "total_zero_variables": total_zero_vars
            },
            "results": results
        }

        # 保存JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
        print(f"{'=' * 80}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="LP文件零变量分析脚本 - 输出可固定为0的变量列表")

    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--lp_input", type=str, required=True, help="LP文件输入路径（文件、目录或包含文件列表的文本文件）")
    parser.add_argument("--topk", type=int, default=1, help="每个prompt生成topk个结果")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["greedy", "sampling"],
                        help="解码方法：greedy（贪婪）或 sampling（采样）")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数（采样时使用）")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p参数（采样时使用）")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大生成token数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "npu", "cpu"],
                        help="设备类型：auto（自动检测）、cuda、npu、cpu")
    parser.add_argument("--num_devices", type=str, default="auto",
                        help="使用的设备数量：auto（自动检测所有可用设备）、数字（指定数量）")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="数据类型：auto（自动检测）、float16、bfloat16、float32")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="是否信任远程代码")
    parser.add_argument("--verbose", action="store_true", help="是否打印详细信息（包括原始模型输出）")
    parser.add_argument("--output_json", type=str, default=None, help="输出JSON文件路径（保存所有结果）")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

