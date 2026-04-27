"""
Draft Model SFT (Supervised Fine-Tuning)
=========================================
Fine-tunes Qwen2.5-Coder-0.5B-Instruct on CodeAlpaca-20k using LoRA (via Unsloth).
This is Task 2 of the speculative-decoding pipeline: aligning the draft model's
distribution to improve acceptance rate during speculative verification.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import trainer_utils
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning of the draft model (Qwen2.5-Coder-0.5B-Instruct)"
    )

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_weights",
        help="Directory to save final LoRA adapter weights.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./sft_checkpoints",
        help="Directory for training checkpoints (used for resuming).",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen2.5-Coder-0.5B-Instruct",
        help="HuggingFace model identifier for the draft model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization (QLoRA).",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the base model with Unsloth and attach a LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    return model, tokenizer


def load_and_format_dataset(tokenizer):
    """
    Load CodeAlpaca-20k and format each example into the Qwen chat template:
        system → "You are a coding assistant."
        user   → instruction (+ optional input)
        assistant → output
    """
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    def format_batch(batch_examples):
        instructions = batch_examples["instruction"]
        inputs = batch_examples["input"]
        outputs = batch_examples["output"]

        processed_samples = []
        for instr, inp, output in zip(instructions, inputs, outputs):
            user_msg = instr
            if inp.strip() != "":
                user_msg = instr + "\n\n" + inp

            messages = [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]

            sample = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            processed_samples.append(sample)

        return {"text": processed_samples}

    dataset = dataset.map(format_batch, batched=True)

    print("=" * 60)
    print("SAMPLE FORMATTED EXAMPLE:")
    print("=" * 60)
    print(dataset[0]["text"])
    print("=" * 60)

    return dataset



def get_resume_checkpoint(checkpoint_dir: str) -> str | None:
    """
    If a previous checkpoint exists in ``checkpoint_dir``, return its path
    so training can be resumed.
    """
    last_checkpoint = trainer_utils.get_last_checkpoint(checkpoint_dir)

    if last_checkpoint is None:
        print("No checkpoint found — starting fresh.")
        return None

    print(f"Resuming from checkpoint: {last_checkpoint}")
    return last_checkpoint



def train(args: argparse.Namespace):
    # --- Model ---
    model, tokenizer = load_model_and_tokenizer(args)

    # --- Dataset ---
    dataset = load_and_format_dataset(tokenizer)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=-1,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            output_dir=args.checkpoint_dir,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
        ),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = get_resume_checkpoint(args.checkpoint_dir)


    trainer.train(resume_from_checkpoint=checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nFinal weights saved to: {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
