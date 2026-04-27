"""
Test: Knowledge-Distilled Draft Model — Inference & Speculative Decoding
========================================================================
Loads the KD-trained draft model (full safetensors weights, NOT LoRA) and
runs three validation stages:

  1. **Standalone generation** — qualitative check of the KD model's output.
  2. **Speculative decoding (KD draft vs base draft)** — measures acceptance
     rate (alpha) improvement from distillation.
  3. **Exact-match verification** — confirms speculative decoding with the KD
     draft still produces token-for-token identical output to the autoregressive
     greedy baseline (mathematical correctness guarantee).

Usage:
    python -m tests.test_kd_inference
    python -m tests.test_kd_inference --kd_weights_dir ./weights/kd_model
    python -m tests.test_kd_inference --kd_weights_dir ./weights/kd_model --prompt "Write a merge sort in Python."
"""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.decoding import autoregressive, speculative


TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_DRAFT_ID   = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
KD_WEIGHTS_DIR  = "./weights/kd_model"

DEVICE = "cuda"
DTYPE  = torch.float16
SYSTEM_PROMPT = "You are a coding assistant."

DEFAULT_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement binary search in Python.",
    "Write a Python function that returns the Fibonacci sequence up to n terms.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference test for KD-trained draft model (full weights)"
    )
    parser.add_argument(
        "--kd_weights_dir", type=str, default=KD_WEIGHTS_DIR,
        help="Path to the KD checkpoint directory containing model.safetensors.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt. If not provided, uses default prompts.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--gamma", type=int, default=5,
        help="Speculation lookahead length.",
    )
    parser.add_argument(
        "--skip_target", action="store_true",
        help="Skip loading the 7B target model (only test standalone generation).",
    )
    return parser.parse_args()

def load_target_model():
    """Load the 7B target model (frozen, fp16)."""
    print(f"  Loading target model: {TARGET_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    return model


def load_draft_model(model_id_or_path: str, label: str = "draft"):
    """Load a draft-sized model from HF hub or local path (full weights)."""
    print(f"  Loading {label} model: {model_id_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    return model, tokenizer

def build_chat_input(tokenizer, user_prompt: str) -> str:
    """Format prompt with the Qwen chat template used during training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens, eos_token_id):
    """Simple greedy generation using model.generate() for quick sanity check."""
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
    )
    return output_ids[0, input_ids.shape[1]:]

def test_standalone_generation(kd_model, tokenizer, prompts, max_new_tokens):
    """Generate responses from the KD model alone to eyeball quality."""
    print("\n" + "=" * 70)
    print("  STAGE 1: Standalone KD Model Generation (qualitative)")
    print("=" * 70)

    eos_token_id = tokenizer.eos_token_id

    for i, user_prompt in enumerate(prompts, 1):
        chat_input = build_chat_input(tokenizer, user_prompt)
        input_ids  = tokenizer(chat_input, return_tensors="pt").input_ids.to(DEVICE)

        print(f"\n{'═' * 70}")
        print(f"  PROMPT {i}: {user_prompt}")
        print(f"{'═' * 70}")

        t0 = time.time()
        gen_ids  = generate_greedy(kd_model, input_ids, max_new_tokens, eos_token_id)
        elapsed  = time.time() - t0
        response = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)

        print(f"\n{'─' * 35} KD MODEL {'─' * 35}")
        print(response)
        print(f"\n  ⏱ {len(gen_ids)} tokens in {elapsed:.2f}s "
              f"({len(gen_ids)/elapsed:.1f} tok/s)")

    print()

def test_speculative_comparison(
    target_model, base_draft, kd_draft, tokenizer,
    prompts, max_new_tokens, gamma,
):
    """
    Run speculative decoding with both the base and KD draft models.
    Compare acceptance rates to quantify the distillation benefit.
    """
    print("\n" + "=" * 70)
    print("  STAGE 2: Speculative Decoding — Base Draft vs KD Draft")
    print("=" * 70)

    eos_token_id = tokenizer.eos_token_id
    results = {"base": [], "kd": []}

    for i, user_prompt in enumerate(prompts, 1):
        chat_input = build_chat_input(tokenizer, user_prompt)
        input_ids  = tokenizer(chat_input, return_tensors="pt").input_ids.to(DEVICE)

        print(f"\n  Prompt {i}: {user_prompt[:60]}…")

        # Base draft
        t0 = time.time()
        _, base_drafted, base_accepted = speculative(
            target_model, base_draft, input_ids, max_new_tokens,
            gamma, eos_token_id, device=DEVICE, greedy=True,
        )
        t_base = time.time() - t0
        base_alpha = base_accepted / base_drafted if base_drafted > 0 else 0.0
        results["base"].append((base_drafted, base_accepted, t_base))

        #  KD draft 
        t0 = time.time()
        _, kd_drafted, kd_accepted = speculative(
            target_model, kd_draft, input_ids, max_new_tokens,
            gamma, eos_token_id, device=DEVICE, greedy=True,
        )
        t_kd = time.time() - t0
        kd_alpha = kd_accepted / kd_drafted if kd_drafted > 0 else 0.0
        results["kd"].append((kd_drafted, kd_accepted, t_kd))

        print(f"    Base draft alpha = {base_alpha:.2%} "
              f"({base_accepted}/{base_drafted}) — {t_base:.2f}s")
        print(f"    KD   draft alpha = {kd_alpha:.2%} "
              f"({kd_accepted}/{kd_drafted}) — {t_kd:.2f}s")
        delta = kd_alpha - base_alpha
        print(f"    Δalpha = {delta:+.2%}")

    # Aggregate
    total_base_d = sum(r[0] for r in results["base"])
    total_base_a = sum(r[1] for r in results["base"])
    total_kd_d   = sum(r[0] for r in results["kd"])
    total_kd_a   = sum(r[1] for r in results["kd"])

    agg_base_alpha = total_base_a / total_base_d if total_base_d > 0 else 0.0
    agg_kd_alpha   = total_kd_a / total_kd_d if total_kd_d > 0 else 0.0

    print(f"\n{'─' * 70}")
    print(f"  AGGREGATE RESULTS ({len(prompts)} prompts)")
    print(f"{'─' * 70}")
    print(f"  Base draft  → alpha = {agg_base_alpha:.2%} "
          f"({total_base_a}/{total_base_d})")
    print(f"  KD   draft  → alpha = {agg_kd_alpha:.2%} "
          f"({total_kd_a}/{total_kd_d})")
    print(f"  Improvement → Δalpha = {agg_kd_alpha - agg_base_alpha:+.2%}")
    print()

    return results


# Stage 3: Exact-match verification

def test_exact_match(target_model, kd_draft, tokenizer, max_new_tokens, gamma):
    """
    Verify speculative decoding with the KD draft produces token-for-token
    identical output to the autoregressive greedy baseline.
    This is the critical correctness guarantee.
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: Exact-Match Verification (KD draft speculative vs AR)")
    print("=" * 70)

    prompt = "Write a Python function to compute the greatest common divisor of two integers."
    print(f"  Prompt: {prompt}")

    chat_input   = build_chat_input(tokenizer, prompt)
    input_ids    = tokenizer(chat_input, return_tensors="pt").input_ids.to(DEVICE)
    eos_token_id = tokenizer.eos_token_id

    # Autoregressive baseline (target model only)
    print("\n  Running autoregressive baseline…")
    t0 = time.time()
    ar_ids = autoregressive(target_model, input_ids, max_new_tokens, eos_token_id)
    t_ar = time.time() - t0
    print(f"  ✔ Done — {len(ar_ids)} tokens in {t_ar:.2f}s "
          f"({len(ar_ids)/t_ar:.1f} tok/s)")

    # Speculative with KD draft
    print(f"\n  Running speculative decoding (KD draft, γ={gamma})…")
    t0 = time.time()
    spec_ids, drafted, accepted = speculative(
        target_model, kd_draft, input_ids, max_new_tokens,
        gamma, eos_token_id, device=DEVICE, greedy=True,
    )
    t_spec = time.time() - t0
    alpha = accepted / drafted if drafted > 0 else 0.0
    speedup = t_ar / t_spec if t_spec > 0 else float("inf")
    print(f"  ✔ Done — {len(spec_ids)} tokens in {t_spec:.2f}s "
          f"({len(spec_ids)/t_spec:.1f} tok/s)")

    # Compare
    min_len = min(len(ar_ids), len(spec_ids))
    token_match  = torch.equal(ar_ids[:min_len], spec_ids[:min_len])
    length_match = len(ar_ids) == len(spec_ids)
    exact_match  = token_match and length_match

    print(f"\n{'─' * 70}")
    print(f"  RESULT:")
    if exact_match:
        print(f"  ✅ EXACT MATCH — {len(spec_ids)} tokens")
    else:
        print(f"  ❌ MISMATCH")
        if not length_match:
            print(f"     Length: AR={len(ar_ids)}, Speculative={len(spec_ids)}")
        for idx in range(min_len):
            if ar_ids[idx] != spec_ids[idx]:
                print(f"     First diff at position {idx}: "
                      f"AR={ar_ids[idx].item()} "
                      f"({tokenizer.decode([ar_ids[idx].item()])!r}), "
                      f"Spec={spec_ids[idx].item()} "
                      f"({tokenizer.decode([spec_ids[idx].item()])!r})")
                break

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'─'*20} {'─'*10}")
    print(f"  {'AR time':<20} {t_ar:>9.2f}s")
    print(f"  {'Spec time':<20} {t_spec:>9.2f}s")
    print(f"  {'Speedup':<20} {speedup:>9.2f}x")
    print(f"  {'Alpha':<20} {alpha:>9.2%}")
    print(f"  {'Drafted':<20} {drafted:>10}")
    print(f"  {'Accepted':<20} {accepted:>10}")

    # Decoded output
    print(f"\n  DECODED OUTPUT (speculative with KD draft):")
    print(f"{'─' * 70}")
    print(tokenizer.decode(spec_ids.tolist(), skip_special_tokens=True))
    print(f"{'─' * 70}")

    assert exact_match, (
        "Speculative decoding with KD draft does NOT match autoregressive "
        "baseline! This should never happen under greedy decoding."
    )
    print("\n  ✅ EXACT-MATCH TEST PASSED\n")
    return exact_match


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("=" * 70)
    print("  KD Draft Model — Inference & Validation Suite")
    print("=" * 70)
    print(f"  KD weights dir : {args.kd_weights_dir}")
    print(f"  Target model   : {TARGET_MODEL_ID}")
    print(f"  Base draft     : {BASE_DRAFT_ID}")
    print(f"  Gamma          : {args.gamma}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Skip target    : {args.skip_target}")
    print()

    # Load KD model
    print("Loading models…")
    kd_model, kd_tokenizer = load_draft_model(args.kd_weights_dir, label="KD draft")

    #  Stage 1: Standalone generation 
    test_standalone_generation(kd_model, kd_tokenizer, prompts, args.max_new_tokens)

    if args.skip_target:
        print("⚠ Skipping Stages 2 & 3 (--skip_target was set).\n")
        return

    # Load target & base draft
    target_model = load_target_model()
    base_draft, _ = load_draft_model(BASE_DRAFT_ID, label="base draft")

    # Use the target tokenizer for speculative decoding (both models share vocab)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    #  Stage 2: Acceptance rate comparison 
    test_speculative_comparison(
        target_model, base_draft, kd_model, tokenizer,
        prompts, args.max_new_tokens, args.gamma,
    )

    #  Stage 3: Exact-match correctness
    test_exact_match(
        target_model, kd_model, tokenizer,
        args.max_new_tokens, args.gamma,
    )

    print("=" * 70)
    print("  ALL STAGES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
