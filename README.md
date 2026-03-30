<span style="font-size:24px;"><span style="font-size:20px;">**EvoOpt-LLM**</span></span>
# Overview
EvoOpt-LLM provides a comprehensive suite of solutions for fine-tuning and applying Large Language Models (LLMs) to operations research (OR) optimization tasks, using Openpangu-7B as the base model. By integrating modules for model fine-tuning, end-to-end automated modeling, generation of new constraints, and variable pruning, it aims to enhance the specialized capabilities of LLMs in the field of operations research.

Detailed information and technical specifications for Openpangu-7B can be found at: https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model/tree/main

# Core Features
Model Fine-tuning: Utilizes efficient fine-tuning techniques such as LoRA to perform specialized training on the Openpangu-7B base model using domain-specific operations research datasets, primarily achieving the following three functions:
- End-to-End Automated Modeling: Automatically generates mathematical models (e.g., linear programming models) for optimization problems from natural language descriptions using the fine-tuned model.
- End-to-End Generation of New Constraints: Automatically generates new constraints for existing linear programming models to extend problem scenarios.
- Variable Pruning: Identifies and safely fixes decision variables with zero values in the model to compress model size and improve solving efficiency.

# Environment Configuration
This project is based on the Openpangu-7B base model and is developed and optimized for the Huawei Ascend AI processor (NPU) environment.

## System Environment
- Operating System: Huawei BMS (Bare Metal Server)
- Firmware/Driver: ≥ 23.0.6
- AI Software Stack: CANN 8.1.rc1

## Core Python Components
- vllm: 0.9.2
- vllm-ascend: 0.9.2rc1
- torch: 2.5.1
- torch_npu: 2.5.1.post1
- opencompass: 0.5.0
- transformers: 4.53.2

# Project Structure
## I. Model Fine-tuning Module
Fine-tunes the model for specific operations research tasks to enhance its professional capabilities in this domain.
```
finetune/
├── train/                          # Fine-tuning training code
│   ├── finetune.py                # Main fine-tuning program
│   ├── arguments.py               # Model configuration and data preprocessing parameters
│   ├── data.py                    # Management of training text datasets
│   ├── train_config.py            # Training configuration management
│   ├── configs/                   # DeepSpeed configuration folder
│   │   ├── stage3_no_offloading_bf16.json
│   │   ├── stage3_no_offloading_fp16.json
│   │   └── stage3_offloading_bf16.json
│   │   └── stage3_offloading_fp16.json
│   └── output/                    # Fine-tuning output
├── dataset/                        # Fine-tuning dataset
```
**1. Modifying Key Parameters in `train_config.py`**

Before starting the fine-tuning process, focus on adjusting the following parameters based on the actual task:
```
"model_name_or_path": Path to the pre-trained model; change to the actual path of the Openpangu-7B base model available locally or on the server.
"lora_rank": Rank for LoRA, controlling the scale of trainable parameters; larger values increase fitting capability but also increase memory usage.
"lora_alpha": Scaling factor for LoRA, typically set to 2 * lora_rank or a similar value.
"lora_dropout": Dropout for LoRA layers to prevent overfitting.
"lora_target_modules": Specifies which modules to inject LoRA into, generally the linear transformations of the attention layers.
"train_dataset_name_or_path": Path to the training dataset.
"max_seq_length": Maximum input sequence length.
"output_dir": Output path for the fine-tuned model.
```
**2. Running `finetune.py`**

After modifying the parameters, execute the following command:

For a single acceleration card:
```
python finetune.py \
--config train_config.py
```

For multiple acceleration cards:
```
torchrun --nproc_per_node=NUMBER_OF_CARDS finetune.py \
--config train_config.py
```
## II. End-to-End Modeling Module
A full-process solution for automated modeling, constraint generation, and code evaluation.
```
end_to_end_modeling/
├── generate.py                    # Basic automated modeling generation
├── generate_finetuned.py          # Generator using the fine-tuned model
├── generate_constraints.py        # New constraint scenario generator
├── execute.py                     # Code execution and evaluation
├── generate_finetuned_example.sh  # Script for fine-tuned model generation example
├── generate_constraints_example.sh # Script for constraint generation example
└── results/                       # Storage for generation results
    ├── generate_results/          # Results from base model automated modeling
    ├── generate_finetuned/        # Results from fine-tuned model automated modeling
└── generate_constraints/      # Results from new constraint scenario generation
```
The end-to-end modeling module includes two parts: **Automated Modeling** and **New Scenario Constraints**.

(1) The **Automated Modeling** module is used to batch-generate results for automated modeling prompts, providing an evaluation module to assess the performance. The core entry script is `generate_finetuned_example.sh`.

1) Key configurable parameters in the script:
```
BASE_MODEL_PATH: Path to the base model.
LORA_PATH: Directory for the fine-tuned LoRA model.
DATASET_NAME: Directory or file path for the automated modeling task dataset.
DATASET_SPLIT: Specifies which part of the dataset to use (train/test/validation).
SAVE_DIR: Directory for model generation results.
TOPK: Number of results generated per prompt.
DECODING_METHOD: Decoding method.
MAX_TOKENS: Maximum number of tokens to generate.
```
2) Once ready, execute the script directly: `bash generate_finetuned_example.sh`

3) To run the evaluation script, use the following command:
```
python execute.py \
  --input_file INPUT_FILE_PATH (automated modeling generation results) \
  --output_file OUTPUT_FILE_PATH \
  --timeout Maximum allowed time for code execution \
  --max_workers Maximum number of parallel execution threads \
  --majority_voting Whether to enable majority voting metrics \
  --question_field Field name for question text in input_file \
  --answer_field Field name for answer text in input_file \
  --numerical_err_tolerance Numerical error tolerance threshold
```
(2) The core entry script for the **New Scenario Constraints** module is `generate_constraints_example.sh`.

1) Key configurable parameters in the script:
```
BASE_MODEL_PATH: Path to the base model.
LORA_PATH: Directory for the fine-tuned LoRA model.
INPUT_PATH: Path to the LP files for adding new constraints.
SAVE_DIR: Directory for model generation results.
TOPK: Number of results generated per prompt.
DECODING_METHOD: Decoding method.
MAX_TOKENS: Maximum number of tokens to generate.
```
2) Once ready, execute the script directly: `bash generate_constraints_example.sh`

## III. Variable Pruning Module
This module performs optimization problem preprocessing by reducing problem scale through variable pruning to improve solving efficiency. The structure is as follows:
```
end_to_end_pruning/
├── analyze_zero_variables.py      # Zero-value variable analysis
├── analyze_zero_variables.sh      # Analysis script
├── evaluate_zero_variables.py     # Pruning effect evaluation
├── evaluate_zero_variables.sh     # Evaluation script
└── results/                       # Pruning results files
```
The end-to-end pruning module consists of two parts: the **Pruning Execution Module** and the **Pruning Evaluation Module**. The execution module uses the fine-tuned Openpangu-7B model to identify variables for pruning in a given LP file, outputting a set of decision variables that can be safely fixed to 0, thereby achieving automated compression and pruning of the MILP model. The evaluation module compares the pruned variables with the ground truth to calculate precision metrics. The core entry script for the variable pruning module is `analyze_zero_variables_example.sh`.

1) Key configurable parameters in the script:
```
BASE_MODEL_PATH: Path to the base model.
LORA_PATH: Directory for the fine-tuned LoRA model.
LP_INPUT: Path to the LP files to be pruned.
OUTPUT_JSON: Path for the generated JSON file with pruning results.
TOPK: Number of results generated per prompt.
DECODING_METHOD: Decoding method.
MAX_TOKENS: Maximum number of tokens to generate.
```
2) Once ready, execute the script directly: `bash analyze_zero_variables_example.sh`

3) Key configurable parameters in the evaluation script:
```
GENERATED_JSON: Path to the generated pruning results.
ANSWER_JSON: Path to the ground truth.
OUTPUT_JSON: Path for the evaluation output file.
```
4) After completing the configuration, call the following command: `bash evaluate_zero_variables_example.sh`
