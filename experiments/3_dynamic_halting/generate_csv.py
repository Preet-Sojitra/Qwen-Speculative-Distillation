import argparse
import csv
import sys
import time

from datasets import load_dataset


# Append the project root so `engine` is importable when running from repo root
sys.path.insert(0, ".")
from engine.decoding import speculative
from utils.load_model import load_models
from config import DATASET_ID, DEVICE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MLP training CSV (entropy, max_prob, accepted) "
                    "from CodeAlpaca-20k using speculative decoding.",
    )
    parser.add_argument(
        "--num_prompts", type=int, default=100,
        help="Number of prompts to process from CodeAlpaca-20k. "
             "Phase 1: 100, Phase 2: 1000–2000.",
    )
    parser.add_argument(
        "--gamma", type=int, default=4,
        help="Speculation length gamma (default: 4).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Maximum new tokens to generate per prompt (default: 128).",
    )
    parser.add_argument(
        "--draft_model_path", type=str, default=None,
        help="Path to a fine-tuned / KD draft model checkpoint (.pt). "
             "If not provided, the baseline 0.5B model is used.",
    )
    parser.add_argument(
        "--output", type=str, default="3_dynamic_halting/draft_features.csv",
        help="Output CSV path (default: 3_dynamic_halting/draft_features.csv).",
    )
    parser.add_argument(
        "--greedy", action="store_true", default=True,
        help="Use greedy decoding (default: True).",
    )
    parser.add_argument(
        "--no-greedy", dest="greedy", action="store_false",
        help="Use stochastic (sampling) decoding.",
    )
    return parser.parse_args()


def load_prompts(num_prompts: int):
    """
    Load prompts from CodeAlpaca-20k.
    Uses the 'instruction' field as the user prompt.
    """
    print(f"[INFO] Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")

    # Take first num_prompts entries
    prompts = []
    for i, example in enumerate(ds):
        if i >= num_prompts:
            break
        # CodeAlpaca-20k has 'instruction', 'input', 'output' columns
        # Use 'instruction' (+ optional 'input') as the prompt
        instruction = example.get("instruction", "")
        inp         = example.get("input", "")
        if inp and inp.strip():
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction
        prompts.append(prompt)

    print(f"[INFO] Loaded {len(prompts)} prompts.")
    return prompts


def main():
    args = parse_args()

    # ── Load models & data ────────────────────────────────────────────────────
    target_model, draft_model, tokenizer = load_models(args.draft_model_path)
    prompts = load_prompts(args.num_prompts)
    eos_token_id = tokenizer.eos_token_id

    # ── Run speculative decoding with logging ─────────────────────────────────
    all_records = []
    total_drafted  = 0
    total_accepted = 0

    print(f"\n{'='*70}")
    print(f"Generating CSV — {len(prompts)} prompts, gamma={args.gamma}, "
          f"max_new_tokens={args.max_new_tokens}, greedy={args.greedy}")
    print(f"{'='*70}\n")

    t_start = time.time()

    for idx, prompt in enumerate(prompts):
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(
            [text], return_tensors="pt",
        ).input_ids.to(DEVICE)

        try:
            gen_ids, drafted, accepted, records = speculative(
                target_model=target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                eos_token_id=eos_token_id,
                device=DEVICE,
                greedy=args.greedy,
                log_features=True,
            )
        except Exception as e:
            print(f"  [WARN] Prompt {idx} failed: {e}")
            continue

        all_records.extend(records)
        total_drafted  += drafted
        total_accepted += accepted

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            alpha   = total_accepted / total_drafted if total_drafted > 0 else 0
            print(
                f"  [{idx+1:>4}/{len(prompts)}]  "
                f"records={len(all_records):>7,}  "
                f"α={alpha:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t_start
    alpha_final = total_accepted / total_drafted if total_drafted > 0 else 0

    # ── Write CSV ─────────────────────────────────────────────────────────────
    print(f"\n[INFO] Writing {len(all_records):,} records to: {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entropy", "max_prob", "accepted"])
        writer.writeheader()
        writer.writerows(all_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    accepted_count = sum(r["accepted"] for r in all_records)
    rejected_count = len(all_records) - accepted_count

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"  Prompts processed  : {len(prompts)}")
    print(f"  Total draft tokens : {len(all_records):,}")
    print(f"  Accepted           : {accepted_count:,} ({100*accepted_count/len(all_records):.1f}%)")
    print(f"  Rejected           : {rejected_count:,} ({100*rejected_count/len(all_records):.1f}%)")
    print(f"  Global alpha       : {alpha_final:.4f}")
    print(f"  Time               : {elapsed:.1f}s")
    print(f"  CSV saved to       : {args.output}")


if __name__ == "__main__":
    main()
