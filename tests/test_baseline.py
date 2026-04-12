import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.decoding import autoregressive, speculative

TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE          = "cuda"
DTYPE           = torch.float16


# ─── Model Loading ───────────────────────────────────────────────────────────

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



# ─── Sanity Check ────────────────────────────────────────────────────────────

def sanity_check(target_model, draft_model, tokenizer, gamma, max_new_tokens):
    prompt   = "explain bubble sort"
    messages = [{"role": "user", "content": prompt}]
    text     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids    = tokenizer([text], return_tensors="pt").input_ids.to(DEVICE)
    eos_token_id = tokenizer.eos_token_id

    print("\n" + "=" * 70)
    print("SANITY CHECK — Greedy Exact-Match Verification")
    print("=" * 70)
    print(f"  Prompt         : {prompt}")
    print(f"  Gamma (gamma)      : {gamma}")
    print(f"  Max new tokens : {max_new_tokens}\n")

    print("Running autoregressive greedy decoding (baseline)…")
    t0          = time.time()
    baseline_ids = autoregressive(
        target_model, input_ids, max_new_tokens, eos_token_id
    )
    t_baseline  = time.time() - t0
    print(f"  ✔ Done — {len(baseline_ids)} tokens in {t_baseline:.2f}s "
          f"({len(baseline_ids)/t_baseline:.1f} tok/s)")

    print(f"\nRunning speculative decoding (greedy, γ={gamma})…")
    t0 = time.time()
    spec_ids, total_drafted, total_accepted = speculative(
        target_model, draft_model, input_ids, max_new_tokens,
        gamma, eos_token_id, device=DEVICE, greedy=True,
    )
    t_spec = time.time() - t0
    print(f"  ✔ Done — {len(spec_ids)} tokens in {t_spec:.2f}s "
          f"({len(spec_ids)/t_spec:.1f} tok/s)")

    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)

    min_len      = min(len(baseline_ids), len(spec_ids))
    match        = torch.equal(baseline_ids[:min_len], spec_ids[:min_len])
    length_match = len(baseline_ids) == len(spec_ids)

    if match and length_match:
        print("  ✅ EXACT MATCH — All tokens identical!")
    else:
        print("  ❌ MISMATCH DETECTED")
        if not length_match:
            print(f"     Length: baseline={len(baseline_ids)}, speculative={len(spec_ids)}")
        for idx in range(min_len):
            if baseline_ids[idx] != spec_ids[idx]:
                print(f"     First diff at position {idx}: "
                      f"baseline={baseline_ids[idx].item()} "
                      f"({tokenizer.decode([baseline_ids[idx].item()])!r}), "
                      f"speculative={spec_ids[idx].item()} "
                      f"({tokenizer.decode([spec_ids[idx].item()])!r})")
                break

    alpha   = total_accepted / total_drafted if total_drafted > 0 else 0.0
    speedup = t_baseline / t_spec if t_spec > 0 else float("inf")

    print(f"\nMETRICS")
    print(f"  Acceptance rate (alpha) : {alpha:.4f}  ({total_accepted}/{total_drafted})")
    print(f"  Baseline time       : {t_baseline:.2f}s")
    print(f"  Speculative time    : {t_spec:.2f}s")
    print(f"  Speedup             : {speedup:.2f}x")

    print(f"\nDECODED OUTPUT (speculative):")
    print("-" * 70)
    print(tokenizer.decode(spec_ids.tolist(), skip_special_tokens=True))
    print("-" * 70)

    assert match and length_match, (
        "Speculative decoding output does NOT match autoregressive baseline!"
    )
    return alpha, speedup

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    gamma = 4
    max_new_tokens = 512

    target_model, draft_model, tokenizer = load_models()
    sanity_check(target_model, draft_model, tokenizer,
                 gamma=gamma, max_new_tokens=max_new_tokens)

if __name__ == "__main__":
    main()