"""
Test: Dynamic-Halting Speculative Decoding — Greedy Exact-Match
================================================================
Verifies that `speculative_dynamic()` produces token-for-token identical output
to the autoregressive greedy baseline, regardless of the halting threshold.

The MLP only controls how many tokens are *drafted* per iteration; it never
changes which tokens are *committed*.  Under greedy mode the output must be
deterministic and equal to standard autoregressive decoding.

Usage:
    python -m tests.test_dynamic_halting
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.decoding import autoregressive, speculative, speculative_dynamic
from engine.halting import load_halting_mlp

TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE          = "cuda"
DTYPE           = torch.float16


def load_models():
    print(f"Loading target model: {TARGET_MODEL_ID}")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    ).eval()

    print(f"Loading draft model:  {DRAFT_MODEL_ID}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    return target_model, draft_model, tokenizer


def _compare(
    name: str,
    baseline_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    tokenizer,
):
    """Print a comparison and return True if exact match."""
    min_len      = min(len(baseline_ids), len(candidate_ids))
    match        = torch.equal(baseline_ids[:min_len], candidate_ids[:min_len])
    length_match = len(baseline_ids) == len(candidate_ids)

    if match and length_match:
        print(f"  ✅ {name}: EXACT MATCH — {len(candidate_ids)} tokens")
    else:
        print(f"  ❌ {name}: MISMATCH")
        if not length_match:
            print(f"     Length: baseline={len(baseline_ids)}, "
                  f"candidate={len(candidate_ids)}")
        for idx in range(min_len):
            if baseline_ids[idx] != candidate_ids[idx]:
                print(f"     First diff at position {idx}: "
                      f"baseline={baseline_ids[idx].item()} "
                      f"({tokenizer.decode([baseline_ids[idx].item()])!r}), "
                      f"candidate={candidate_ids[idx].item()} "
                      f"({tokenizer.decode([candidate_ids[idx].item()])!r})")
                break

    return match and length_match


def test_dynamic_halting(target_model, draft_model, tokenizer,
                         max_gamma, max_new_tokens):
    prompt   = "explain bubble sort"
    messages = [{"role": "user", "content": prompt}]
    text     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids    = tokenizer([text], return_tensors="pt").input_ids.to(DEVICE)
    eos_token_id = tokenizer.eos_token_id

    print("\n" + "=" * 70)
    print("TEST: Dynamic Halting — Greedy Exact-Match Verification")
    print("=" * 70)
    print(f"  Prompt         : {prompt}")
    print(f"  Max gamma      : {max_gamma}")
    print(f"  Max new tokens : {max_new_tokens}\n")

    # 1. Autoregressive baseline
    print("Running autoregressive greedy decoding (baseline)…")
    t0 = time.time()
    baseline_ids = autoregressive(
        target_model, input_ids, max_new_tokens, eos_token_id
    )
    t_baseline = time.time() - t0
    print(f"  ✔ Done — {len(baseline_ids)} tokens in {t_baseline:.2f}s "
          f"({len(baseline_ids)/t_baseline:.1f} tok/s)")

    # 2. Standard speculative (fixed gamma, for reference)
    print(f"\nRunning standard speculative decoding (γ={max_gamma})…")
    t0 = time.time()
    spec_ids, spec_drafted, spec_accepted = speculative(
        target_model, draft_model, input_ids, max_new_tokens,
        max_gamma, eos_token_id, device=DEVICE, greedy=True,
    )
    t_spec = time.time() - t0
    spec_alpha = spec_accepted / spec_drafted if spec_drafted > 0 else 0.0
    print(f"  ✔ Done — {len(spec_ids)} tokens in {t_spec:.2f}s "
          f"({len(spec_ids)/t_spec:.1f} tok/s)")

    # 3. Dynamic halting: "always accept" (should behave like fixed gamma)
    print(f"\nRunning dynamic halting — always-accept MLP (threshold=0.0)…")

    # A dummy predict_fn that always returns 1.0 (never halts)
    always_accept_fn = lambda entropy, max_prob: 1.0

    t0 = time.time()
    dyn_always_ids, dyn_always_drafted, dyn_always_accepted, gammas_always = \
        speculative_dynamic(
            target_model, draft_model, input_ids, max_new_tokens,
            max_gamma=max_gamma,
            min_gamma=1,
            halt_threshold=0.0,     # never halts (accept_prob=1.0 ≥ 0.0)
            halting_predict_fn=always_accept_fn,
            eos_token_id=eos_token_id,
            device=DEVICE,
            greedy=True,
        )
    t_dyn_always = time.time() - t0
    print(f"  ✔ Done — {len(dyn_always_ids)} tokens in {t_dyn_always:.2f}s")

    # 4. Dynamic halting: "always reject" (halts after min_gamma every time)
    print(f"\nRunning dynamic halting — always-reject MLP (threshold=1.0)…")

    always_reject_fn = lambda entropy, max_prob: 0.0

    t0 = time.time()
    dyn_reject_ids, dyn_reject_drafted, dyn_reject_accepted, gammas_reject = \
        speculative_dynamic(
            target_model, draft_model, input_ids, max_new_tokens,
            max_gamma=max_gamma,
            min_gamma=1,
            halt_threshold=1.0,     # always halts (accept_prob=0.0 < 1.0)
            halting_predict_fn=always_reject_fn,
            eos_token_id=eos_token_id,
            device=DEVICE,
            greedy=True,
        )
    t_dyn_reject = time.time() - t0
    print(f"  ✔ Done — {len(dyn_reject_ids)} tokens in {t_dyn_reject:.2f}s")

    # 5. Dynamic halting: trained MLP with real weights
    print(f"\nRunning dynamic halting — trained MLP (threshold=0.5)…")

    halting_fn = load_halting_mlp(device=DEVICE)

    t0 = time.time()
    dyn_mlp_ids, dyn_mlp_drafted, dyn_mlp_accepted, gammas_mlp = \
        speculative_dynamic(
            target_model, draft_model, input_ids, max_new_tokens,
            max_gamma=max_gamma,
            min_gamma=1,
            halt_threshold=0.5,
            halting_predict_fn=halting_fn,
            eos_token_id=eos_token_id,
            device=DEVICE,
            greedy=True,
        )
    t_dyn_mlp = time.time() - t0
    dyn_alpha = dyn_mlp_accepted / dyn_mlp_drafted if dyn_mlp_drafted > 0 else 0.0
    print(f"  ✔ Done — {len(dyn_mlp_ids)} tokens in {t_dyn_mlp:.2f}s")

    # Comparisons
    print("\n" + "-" * 70)
    print("COMPARISONS (all vs autoregressive baseline)")
    print("-" * 70)

    ok_spec       = _compare("Standard speculative",    baseline_ids, spec_ids, tokenizer)
    ok_always     = _compare("Dynamic (always accept)", baseline_ids, dyn_always_ids, tokenizer)
    ok_reject     = _compare("Dynamic (always reject)", baseline_ids, dyn_reject_ids, tokenizer)
    ok_mlp        = _compare("Dynamic (trained MLP)",   baseline_ids, dyn_mlp_ids, tokenizer)

    # Metrics
    print("\n" + "-" * 70)
    print("METRICS")
    print("-" * 70)

    print(f"\n  {'Variant':<25} {'Time':>7} {'Speedup':>8} {'Alpha':>7} {'Drafted':>8} {'Accepted':>9}")
    print(f"  {'─'*25} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")

    def _row(name, t, drafted, accepted):
        alpha   = accepted / drafted if drafted > 0 else 0.0
        speedup = t_baseline / t if t > 0 else float("inf")
        print(f"  {name:<25} {t:>6.2f}s {speedup:>7.2f}x {alpha:>6.2%} {drafted:>8} {accepted:>9}")

    _row("Autoregressive",        t_baseline, len(baseline_ids), len(baseline_ids))
    _row("Standard speculative",  t_spec,     spec_drafted,       spec_accepted)
    _row("Dynamic (always accept)", t_dyn_always, dyn_always_drafted, dyn_always_accepted)
    _row("Dynamic (always reject)", t_dyn_reject, dyn_reject_drafted, dyn_reject_accepted)
    _row("Dynamic (trained MLP)",   t_dyn_mlp,    dyn_mlp_drafted,    dyn_mlp_accepted)

    # Dynamic gamma distribution
    print(f"\n  Gamma distribution (trained MLP):")
    if gammas_mlp:
        avg_gamma = sum(gammas_mlp) / len(gammas_mlp)
        print(f"    Iterations   : {len(gammas_mlp)}")
        print(f"    Avg gamma    : {avg_gamma:.2f}")
        print(f"    Min gamma    : {min(gammas_mlp)}")
        print(f"    Max gamma    : {max(gammas_mlp)}")
        # Histogram
        from collections import Counter
        hist = Counter(gammas_mlp)
        print(f"    Distribution : {dict(sorted(hist.items()))}")

    # Decoded output
    print(f"\nDECODED OUTPUT (dynamic halting, trained MLP):")
    print("-" * 70)
    print(tokenizer.decode(dyn_mlp_ids.tolist(), skip_special_tokens=True))
    print("-" * 70)

    # Assertion
    all_ok = ok_spec and ok_always and ok_reject and ok_mlp
    assert all_ok, (
        "Dynamic halting speculative decoding output does NOT match "
        "autoregressive baseline!"
    )
    print("\n ALL TESTS PASSED — Dynamic halting preserves exact greedy output.\n")
    return all_ok


def main():
    max_gamma      = 5
    max_new_tokens = 512

    target_model, draft_model, tokenizer = load_models()
    test_dynamic_halting(
        target_model, draft_model, tokenizer,
        max_gamma=max_gamma,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    main()
