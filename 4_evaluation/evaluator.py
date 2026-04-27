"""
4_evaluation/evaluator.py
─────────────────────────
Full evaluation script that produces two report tables:

  Table 1 — Alignment Ablation  (Baseline vs SFT vs KD)
  Table 2 — Dynamic Lookahead   (Fixed γ vs Dynamic γ on the KD draft)

Run from the project root:
    python -m 4_evaluation.evaluator [OPTIONS]

Requirements:
  - Target model:  Qwen/Qwen2.5-Coder-7B-Instruct
  - Draft model:   Qwen/Qwen2.5-Coder-0.5B-Instruct  (off-the-shelf, SFT, KD)
  - HumanEval dataset (openai/openai_humaneval)
  - Trained halting MLP weights (3_dynamic_halting/mlp_weights.pt)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TARGET_MODEL_ID, DRAFT_MODEL_ID, DEVICE, DTYPE
from engine.decoding import autoregressive, speculative, speculative_dynamic
from engine.halting import load_halting_mlp


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_NUM_PROMPTS   = 20
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_GAMMA          = 4
DEFAULT_MAX_GAMMA      = 6
DEFAULT_MIN_GAMMA      = 1
DEFAULT_HALT_THRESHOLD = 0.5


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate speculative decoding: alignment ablation + dynamic halting."
    )
    p.add_argument("--num_prompts", type=int, default=DEFAULT_NUM_PROMPTS,
                   help="Number of HumanEval prompts to evaluate.")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Max new tokens per prompt.")
    p.add_argument("--gamma", type=int, default=DEFAULT_GAMMA,
                   help="Fixed gamma for Table 1.")
    p.add_argument("--max_gamma", type=int, default=DEFAULT_MAX_GAMMA,
                   help="Max gamma for dynamic halting (Table 2).")
    p.add_argument("--min_gamma", type=int, default=DEFAULT_MIN_GAMMA,
                   help="Min gamma for dynamic halting (Table 2).")
    p.add_argument("--halt_threshold", type=float, default=DEFAULT_HALT_THRESHOLD,
                   help="MLP halt threshold (Table 2).")
    # ── Draft model checkpoint paths (set to None to skip that row) ───────────
    p.add_argument("--sft_draft_path", type=str, default=None,
                   help="Path to LoRA-SFT draft checkpoint (.pt). "
                        "Leave empty to skip the SFT row.")
    p.add_argument("--kd_draft_path", type=str, default=None,
                   help="Path to KD draft checkpoint (.pt). "
                        "Leave empty to skip KD rows.")
    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).parent / "results"),
                   help="Directory to write result JSONs.")
    return p.parse_args()


# ─── Model loading helpers ────────────────────────────────────────────────────

def load_target_model():
    """Load the 7B target model (shared across all runs)."""
    print(f"[INFO] Loading target model: {TARGET_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()
    return model


def load_draft_model(checkpoint_path=None):
    """
    Load a draft model.
    - checkpoint_path=None  → off-the-shelf 0.5B
    - checkpoint_path=<path> → load base 0.5B then overlay checkpoint weights
    """
    if checkpoint_path is not None:
        print(f"[INFO] Loading draft model with checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
        ).eval()
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Draft checkpoint weights loaded.")
    else:
        print(f"[INFO] Loading off-the-shelf draft model: {DRAFT_MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
        ).eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TARGET_MODEL_ID)


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_prompts(num_prompts: int):
    """Return list[str] of HumanEval coding prompts."""
    print(f"[INFO] Loading openai_humaneval (first {num_prompts} prompts)…")
    ds = load_dataset("openai/openai_humaneval", split="test")
    ds = ds.select(range(min(num_prompts, len(ds))))
    return [item["prompt"] for item in ds]


# ─── Evaluation runners ──────────────────────────────────────────────────────

def run_autoregressive(target_model, tokenizer, prompts, max_new_tokens):
    """
    Baseline: pure autoregressive decoding with the target model.
    Returns dict with aggregate metrics.
    """
    print("\n[RUN] Autoregressive baseline (target-only)…")
    total_tokens = 0
    total_time   = 0.0
    eos_id = tokenizer.eos_token_id

    for prompt in tqdm(prompts, desc="Autoregressive"):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        t0 = time.time()
        gen = autoregressive(target_model, input_ids, max_new_tokens, eos_id)
        t1 = time.time()
        total_tokens += len(gen)
        total_time   += (t1 - t0)

    tps = total_tokens / total_time if total_time > 0 else 0
    print(f"  → {total_tokens} tokens in {total_time:.2f}s  ({tps:.1f} tok/s)")
    return {
        "total_tokens": total_tokens,
        "total_time":   total_time,
        "tps":          tps,
    }


def run_speculative_fixed(
    target_model, draft_model, tokenizer, prompts, max_new_tokens, gamma,
    label="speculative",
):
    """
    Speculative decoding with fixed gamma.
    Returns dict with aggregate metrics (alpha, tokens/step, tps, etc.).
    """
    print(f"\n[RUN] {label}  (fixed γ={gamma})…")
    total_tokens   = 0
    total_time     = 0.0
    total_drafted  = 0
    total_accepted = 0
    total_steps    = 0          # number of verify iterations
    total_draft_fwd = 0         # total draft forward passes (= total_drafted)
    eos_id = tokenizer.eos_token_id

    for prompt in tqdm(prompts, desc=label):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        t0 = time.time()
        gen, drafted, accepted = speculative(
            target_model, draft_model, input_ids, max_new_tokens,
            gamma=gamma, eos_token_id=eos_id, device=DEVICE, greedy=True,
        )
        t1 = time.time()
        total_tokens   += len(gen)
        total_time     += (t1 - t0)
        total_drafted  += drafted
        total_accepted += accepted
        # Each verify iteration drafts exactly `gamma` tokens (or less at the end)
        # so #steps ≈ drafted / gamma
        total_steps    += (drafted + gamma - 1) // gamma
        total_draft_fwd += drafted

    alpha       = total_accepted / total_drafted if total_drafted > 0 else 0
    tokens_step = total_tokens / total_steps if total_steps > 0 else 0
    tps         = total_tokens / total_time  if total_time  > 0 else 0
    wasted      = total_drafted - total_accepted

    print(f"  → α={alpha:.4f}  tok/step={tokens_step:.2f}  "
          f"TPS={tps:.1f}  wasted={wasted}")
    return {
        "total_tokens":   total_tokens,
        "total_time":     total_time,
        "tps":            tps,
        "alpha":          alpha,
        "tokens_per_step": tokens_step,
        "total_drafted":  total_drafted,
        "total_accepted": total_accepted,
        "total_steps":    total_steps,
        "wasted_tokens":  wasted,
        "draft_fwd_passes": total_draft_fwd,
    }


def run_speculative_dynamic(
    target_model, draft_model, tokenizer, prompts, max_new_tokens,
    max_gamma, min_gamma, halt_threshold, halting_predict_fn,
    label="dynamic",
):
    """
    Speculative decoding with dynamic gamma (MLP halting).
    Returns dict with aggregate metrics.
    """
    print(f"\n[RUN] {label}  (γ ∈ [{min_gamma}, {max_gamma}], thresh={halt_threshold})…")
    total_tokens   = 0
    total_time     = 0.0
    total_drafted  = 0
    total_accepted = 0
    total_steps    = 0
    total_draft_fwd = 0
    all_gammas     = []
    eos_id = tokenizer.eos_token_id

    for prompt in tqdm(prompts, desc=label):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        t0 = time.time()
        gen, drafted, accepted, gammas = speculative_dynamic(
            target_model, draft_model, input_ids, max_new_tokens,
            max_gamma=max_gamma, min_gamma=min_gamma,
            halt_threshold=halt_threshold,
            halting_predict_fn=halting_predict_fn,
            eos_token_id=eos_id, device=DEVICE, greedy=True,
        )
        t1 = time.time()
        total_tokens   += len(gen)
        total_time     += (t1 - t0)
        total_drafted  += drafted
        total_accepted += accepted
        total_steps    += len(gammas)
        total_draft_fwd += drafted
        all_gammas.extend(gammas)

    alpha       = total_accepted / total_drafted if total_drafted > 0 else 0
    tokens_step = total_tokens / total_steps if total_steps > 0 else 0
    tps         = total_tokens / total_time  if total_time  > 0 else 0
    wasted      = total_drafted - total_accepted
    avg_gamma   = sum(all_gammas) / len(all_gammas) if all_gammas else 0

    print(f"  → α={alpha:.4f}  avg_γ={avg_gamma:.2f}  tok/step={tokens_step:.2f}  "
          f"TPS={tps:.1f}  wasted={wasted}")
    return {
        "total_tokens":    total_tokens,
        "total_time":      total_time,
        "tps":             tps,
        "alpha":           alpha,
        "tokens_per_step": tokens_step,
        "total_drafted":   total_drafted,
        "total_accepted":  total_accepted,
        "total_steps":     total_steps,
        "wasted_tokens":   wasted,
        "draft_fwd_passes": total_draft_fwd,
        "avg_gamma":       avg_gamma,
        "gamma_histogram": {str(g): all_gammas.count(g) for g in set(all_gammas)},
    }


# ─── Pretty-print tables ─────────────────────────────────────────────────────

def print_table_1(rows: list[dict], ar_metrics: dict):
    """
    Table 1 — Alignment Ablation
    Columns: Draft Model | α | Tokens/Step | Wall-Clock Speedup
    """
    ar_time = ar_metrics["total_time"]

    hdr  = f"{'Draft Model':<25} {'α':>8} {'Tok/Step':>10} {'Speedup':>10}"
    line = "─" * len(hdr)
    print(f"\n{'═'*len(hdr)}")
    print("TABLE 1 — Alignment Ablation (Baseline vs SFT vs KD)")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print(line)
    for r in rows:
        speedup = ar_time / r["total_time"] if r["total_time"] > 0 else 0
        print(f"{r['label']:<25} {r['alpha']:>8.4f} "
              f"{r['tokens_per_step']:>10.2f} {speedup:>9.2f}x")
    print(line)
    print(f"  (Autoregressive baseline: {ar_metrics['tps']:.1f} tok/s, "
          f"{ar_time:.2f}s total)\n")


def print_table_2(rows: list[dict]):
    """
    Table 2 — Dynamic Lookahead Efficiency
    Columns: Strategy | Draft FWD Passes | Wasted Tokens | Final TPS
    """
    hdr  = f"{'Strategy':<35} {'Draft FWDs':>12} {'Wasted Tok':>12} {'TPS':>8}"
    line = "─" * len(hdr)
    print(f"\n{'═'*len(hdr)}")
    print("TABLE 2 — Dynamic Lookahead Efficiency (Fixed γ vs Dynamic γ)")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print(line)
    for r in rows:
        print(f"{r['label']:<35} {r['draft_fwd_passes']:>12} "
              f"{r['wasted_tokens']:>12} {r['tps']:>8.1f}")
    print(line)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    prompts   = load_prompts(args.num_prompts)

    # ══════════════════════════════════════════════════════════════════════════
    # Load the target model (shared across all experiments)
    # ══════════════════════════════════════════════════════════════════════════
    target_model = load_target_model()

    # ══════════════════════════════════════════════════════════════════════════
    # 0) Autoregressive baseline (target-only, no draft)
    # ══════════════════════════════════════════════════════════════════════════
    ar_metrics = run_autoregressive(
        target_model, tokenizer, prompts, args.max_new_tokens,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1 — Alignment Ablation
    # ══════════════════════════════════════════════════════════════════════════
    table1_rows = []

    # Row 1: Off-the-shelf draft (always run)
    draft_baseline = load_draft_model(checkpoint_path=None)
    m = run_speculative_fixed(
        target_model, draft_baseline, tokenizer, prompts,
        args.max_new_tokens, args.gamma,
        label="Off-the-shelf 0.5B",
    )
    m["label"] = "Off-the-shelf 0.5B"
    table1_rows.append(m)
    del draft_baseline
    torch.cuda.empty_cache()

    # Row 2: LoRA SFT draft (skip if path not provided)
    if args.sft_draft_path:
        draft_sft = load_draft_model(checkpoint_path=args.sft_draft_path)
        m = run_speculative_fixed(
            target_model, draft_sft, tokenizer, prompts,
            args.max_new_tokens, args.gamma,
            label="LoRA SFT 0.5B",
        )
        m["label"] = "LoRA SFT 0.5B"
        table1_rows.append(m)
        del draft_sft
        torch.cuda.empty_cache()
    else:
        print("\n[SKIP] SFT draft row — no --sft_draft_path provided.")

    # Row 3: KD draft (skip if path not provided)
    kd_metrics = None
    draft_kd   = None
    if args.kd_draft_path:
        draft_kd = load_draft_model(checkpoint_path=args.kd_draft_path)
        m = run_speculative_fixed(
            target_model, draft_kd, tokenizer, prompts,
            args.max_new_tokens, args.gamma,
            label="KD 0.5B",
        )
        m["label"] = "KD 0.5B"
        kd_metrics = m
        table1_rows.append(m)
        # Keep draft_kd alive for Table 2
    else:
        print("\n[SKIP] KD draft row — no --kd_draft_path provided.")

    print_table_1(table1_rows, ar_metrics)

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2 — Dynamic Lookahead Efficiency
    # ══════════════════════════════════════════════════════════════════════════
    table2_rows = []

    # Determine which draft model to use for Table 2:
    # prefer KD draft if available, otherwise fall back to off-the-shelf
    if draft_kd is not None:
        draft_for_t2 = draft_kd
        t2_draft_label = "KD 0.5B"
    else:
        print("\n[INFO] No KD draft available — using off-the-shelf for Table 2.")
        draft_for_t2 = load_draft_model(checkpoint_path=None)
        t2_draft_label = "Off-the-shelf 0.5B"

    # Row 1: Fixed gamma (reuse KD metrics if available, else re-run)
    if kd_metrics is not None:
        fixed_m = kd_metrics
    else:
        fixed_m = run_speculative_fixed(
            target_model, draft_for_t2, tokenizer, prompts,
            args.max_new_tokens, args.gamma,
            label=f"{t2_draft_label} (fixed γ={args.gamma})",
        )
    fixed_m["label"] = f"{t2_draft_label} (fixed γ={args.gamma})"
    table2_rows.append(fixed_m)

    # Row 2: Dynamic gamma with MLP halting
    halting_fn = load_halting_mlp(device="cpu")   # tiny MLP, CPU is fine
    dyn_m = run_speculative_dynamic(
        target_model, draft_for_t2, tokenizer, prompts, args.max_new_tokens,
        max_gamma=args.max_gamma, min_gamma=args.min_gamma,
        halt_threshold=args.halt_threshold,
        halting_predict_fn=halting_fn,
        label=f"{t2_draft_label} + Dynamic MLP",
    )
    dyn_m["label"] = f"{t2_draft_label} + Dynamic MLP"
    table2_rows.append(dyn_m)

    print_table_2(table2_rows)

    # ══════════════════════════════════════════════════════════════════════════
    # Save all results to JSON
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {
        "config": {
            "target_model":   TARGET_MODEL_ID,
            "draft_model":    DRAFT_MODEL_ID,
            "num_prompts":    args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gamma":          args.gamma,
            "max_gamma":      args.max_gamma,
            "min_gamma":      args.min_gamma,
            "halt_threshold": args.halt_threshold,
            "sft_draft_path": args.sft_draft_path,
            "kd_draft_path":  args.kd_draft_path,
        },
        "autoregressive": ar_metrics,
        "table1_alignment_ablation": table1_rows,
        "table2_dynamic_lookahead":  table2_rows,
    }

    out_path = output_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[INFO] Full results saved to {out_path}")


if __name__ == "__main__":
    main()
