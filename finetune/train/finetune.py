#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
from typing import Optional
from functools import partial
import datasets
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from arguments import ModelArguments, DataArguments
from data import CustomDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Trainer to fix torch_npu compatibility issue
class NPUCompatibleTrainer(Trainer):
    """Custom Trainer that disables memory tracking to avoid torch_npu decorator conflicts"""

    def __init__(self, *args, **kwargs):
        # Disable memory tracking for NPU compatibility
        if 'args' in kwargs:
            kwargs['args'].skip_memory_metrics = True
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to ensure model is in training mode
        and handle gradient checkpointing properly with LoRA
        """
        model.train()

        # Ensure LoRA parameters require gradients in distributed setting
        if hasattr(model, 'module'):
            # DDP wrapped model
            base_model = model.module
        else:
            base_model = model

        # For PEFT models, ensure training mode is properly set
        if hasattr(base_model, 'base_model'):
            base_model.base_model.train()

        return super().training_step(model, inputs, num_items_in_batch)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize distributed training for NPU if needed
    # TrainingArguments will handle DDP setup, but we need to initialize NPU backend
    if training_args.local_rank != -1:
        # Distributed training mode
        local_rank = training_args.local_rank
        rank = int(os.environ.get('RANK', training_args.local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', training_args.world_size))

        # Initialize NPU distributed backend if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(
                backend='hccl',  # Huawei Collective Communication Library for NPU
                init_method='env://',
                rank=rank,
                world_size=world_size
            )

        # Set device for current process
        torch.npu.set_device(local_rank)
        device = torch.device(f'npu:{local_rank}')
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        # Single device training mode
        device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')
        logger.info("Single device training mode")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")

    # Move model to NPU device for distributed training
    if training_args.local_rank != -1:
        # For distributed training, model will be moved to device by Trainer
        # But we ensure it's on the correct device here
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
    elif torch.npu.is_available():
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0,
                                    1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            logging.info("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        logger.info("Initializing LORA model...")

        # Important: Enable gradient checkpointing BEFORE applying LoRA
        # This ensures proper setup for distributed training
        if training_args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing before LoRA initialization")
            model.gradient_checkpointing_enable()
            # For some models, we need to enable input to require gradients
            model.enable_input_require_grads()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        logger.info(f"LoraConfig: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        # For full fine-tuning, enable all parameters
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    # set up datasets
    train_dataset = CustomDataset(training_args, data_args, model_args, tokenizer)

    # Verify trainable parameters before training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Check LoRA configuration and model setup.")

    # Log parameter details for debugging
    if training_args.local_rank in [-1, 0]:  # Only log on main process
        logger.info("Parameter gradient status:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: requires_grad=True, shape={param.shape}")

    # Ensure distributed training is properly configured
    if training_args.local_rank != -1:
        # Set ddp_find_unused_parameters to False for better performance with LoRA
        if not hasattr(training_args, 'ddp_find_unused_parameters'):
            training_args.ddp_find_unused_parameters = False
        logger.info(f"Distributed training enabled: local_rank={training_args.local_rank}, "
                    f"world_size={training_args.world_size if hasattr(training_args, 'world_size') else 'N/A'}")

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    # Use NPUCompatibleTrainer to avoid memory tracking issues with torch_npu
    trainer = NPUCompatibleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Only save on main process to avoid conflicts
    if training_args.local_rank in [-1, 0]:
        trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")

if __name__ == "__main__":
    main()
