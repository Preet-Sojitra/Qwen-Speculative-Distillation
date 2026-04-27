"""
Quick inference test for the SFT fine-tuned draft model.
Loads the base Qwen2.5-Coder-0.5B-Instruct + LoRA adapter from sft_weights/
and generates a response to compare against the base model.

Usage:
    python test_sft_inference.py --adapter_dir ./sft_weights
    python test_sft_inference.py --adapter_dir ./sft_weights --prompt "Write a Python function to merge two sorted lists."
"""

import argparse
import torch
from unsloth import FastLanguageModel


BASE_MODEL = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = "You are a coding assistant."

DEFAULT_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement binary search in Python.",
    "Write a Python function that returns the Fibonacci sequence up to n terms.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test SFT fine-tuned draft model")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./sft_weights",
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to test. If not provided, uses default prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        default=True,
        help="Also generate with the base model for comparison.",
    )
    return parser.parse_args()


def build_chat_input(tokenizer, user_prompt: str) -> str:
    """Format prompt using the same Qwen chat template used during training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant prefix so model generates
    )


def generate_response(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    """Tokenize, generate, and decode a single response."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic
            temperature=1.0,
            use_cache=True,
        )
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    # ── Load fine-tuned model (base + LoRA adapter) ──
    print("=" * 70)
    print("Loading FINE-TUNED model (base + LoRA adapter)...")
    print("=" * 70)
    ft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(ft_model)

    # ── Optionally load base model for comparison ──
    base_model = None
    if args.compare_base:
        print("\nLoading BASE model (no fine-tuning)...")
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(base_model)

    # ── Generate ──
    for i, user_prompt in enumerate(prompts, 1):
        chat_input = build_chat_input(tokenizer, user_prompt)

        print("\n" + "═" * 70)
        print(f"  PROMPT {i}: {user_prompt}")
        print("═" * 70)

        # Fine-tuned
        ft_response = generate_response(ft_model, tokenizer, chat_input, args.max_new_tokens)
        print(f"\n{'─'*35} FINE-TUNED {'─'*35}")
        print(ft_response)

        # Base
        if base_model is not None:
            base_response = generate_response(base_model, tokenizer, chat_input, args.max_new_tokens)
            print(f"\n{'─'*35} BASE MODEL {'─'*35}")
            print(base_response)

        print()

    print("═" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
