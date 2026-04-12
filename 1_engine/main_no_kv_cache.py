"""
Speculative Decoding Inference Engine
=====================================
Implements Algorithm 1 from "Fast Inference from Transformers via Speculative
Decoding" (Leviathan et al., 2023).

NO KV-cache — every forward pass reprocesses the full sequence.
This is slower but correct and simple to verify.

Models:
    Target (M_p): Qwen/Qwen2.5-Coder-1.5B-Instruct
    Draft  (M_q): Qwen/Qwen2.5-Coder-0.5B-Instruct
"""

import time
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE          = "cuda"
DTYPE           = torch.float16


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_models():
    print(f"Loading target model: {TARGET_MODEL_ID}")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()

    print(f"Loading draft model:  {DRAFT_MODEL_ID}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    return target_model, draft_model, tokenizer


# ─── Autoregressive Greedy Decoding (Baseline, no KV-cache) ─────────────────

@torch.no_grad()
def autoregressive_greedy_decode(model, input_ids, max_new_tokens, eos_token_id):
    """
    Standard greedy decoding. Each step feeds the FULL sequence so far.
    Ground truth for the sanity check.
    """
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        logits    = model(generated_ids).logits          # (1, seq, vocab)
        next_tok  = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
        generated_ids = torch.cat([generated_ids, next_tok], dim=-1)
        if next_tok.item() == eos_token_id:
            break

    # Return only the newly generated part (strip the prompt)
    return generated_ids[0, input_ids.shape[1]:]


# ─── Speculative Decoding (Algorithm 1, no KV-cache) ────────────────────────

@torch.no_grad()
def speculative_decode(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens,
    gamma,
    eos_token_id,
    greedy=True,
):
    """
    Speculative Decoding — Algorithm 1 from Leviathan et al. (2023).
    No KV-cache: every forward pass reprocesses the full sequence.

    Returns:
        generated_ids         — 1-D tensor of new token IDs (no prompt)
        total_draft_tokens    — total tokens proposed by the draft model
        total_accepted_tokens — draft tokens accepted by the target model
    """
    # current_ids holds the full sequence: prompt + all committed tokens so far
    current_ids = input_ids.clone()

    total_draft_tokens    = 0
    total_accepted_tokens = 0

    while True:
        if current_ids.shape[1] - input_ids.shape[1] >= max_new_tokens:
            break

        current_gamma = min(gamma, max_new_tokens - (current_ids.shape[1] - input_ids.shape[1]))

        # ── STEP 1: Draft γ tokens from M_q ──────────────────────────────────
        draft_tokens = []   # x_1 … x_γ
        draft_probs  = []   # q(· | context) at each draft step

        draft_ids = current_ids.clone()
        for _ in range(current_gamma):
            q_logits = draft_model(draft_ids).logits          # full sequence
            q        = F.softmax(q_logits[:, -1, :], dim=-1)  # last-position probs
            if greedy:
                x = torch.argmax(q, dim=-1)                   # (1,)
            else:
                x = torch.multinomial(q, num_samples=1).squeeze(-1)
            draft_tokens.append(x.item())
            draft_probs.append(q)
            draft_ids = torch.cat([draft_ids, x.view(1, 1)], dim=-1)

        total_draft_tokens += current_gamma

        # ── STEP 2: Verify all γ drafts with M_p in ONE forward pass ─────────
        # Feed current_ids + all draft tokens together.
        # logits[:, prefix_len - 1 + j, :] is p(· | current + draft[:j])
        # which is used to accept/reject draft_tokens[j].
        draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=DEVICE)
        verify_ids   = torch.cat([current_ids, draft_tensor], dim=-1)

        p_logits = target_model(verify_ids).logits   # (1, prefix+γ, vocab)
        prefix_len = current_ids.shape[1]            # length before drafts

        # ── STEP 3: Accept / Reject ───────────────────────────────────────────
        n = 0  # number of accepted draft tokens
        if greedy:
            for j in range(current_gamma):
                # logit at position (prefix_len - 1 + j) gives p(· | ctx + draft[:j])
                p_j = F.softmax(p_logits[:, prefix_len - 1 + j, :], dim=-1)
                if torch.argmax(p_j, dim=-1).item() == draft_tokens[j]:
                    n += 1
                else:
                    break
        else:
            for j in range(current_gamma):
                p_j = F.softmax(p_logits[:, prefix_len - 1 + j, :], dim=-1)
                q_j = draft_probs[j]
                x_j = draft_tokens[j]
                r   = torch.rand(1, device=DEVICE)
                if r.item() < min(1.0, (p_j[0, x_j] / q_j[0, x_j]).item()):
                    n += 1
                else:
                    break

        total_accepted_tokens += n

        # ── STEP 4: Bonus token ───────────────────────────────────────────────
        # p_{n+1}: distribution after accepting the first n draft tokens.
        p_bonus = F.softmax(p_logits[:, prefix_len - 1 + n, :], dim=-1)

        if greedy:
            bonus_token = torch.argmax(p_bonus, dim=-1).item()
        else:
            if n < current_gamma:
                # Adjusted distribution: normalise max(0, p - q)
                q_n      = draft_probs[n]
                adjusted = torch.clamp(p_bonus - q_n, min=0.0)
                adj_sum  = adjusted.sum()
                adjusted = adjusted / adj_sum if adj_sum > 1e-10 else p_bonus
                bonus_token = torch.multinomial(adjusted, num_samples=1).item()
            else:
                bonus_token = torch.multinomial(p_bonus, num_samples=1).item()

        # ── STEP 5: Commit and check EOS ─────────────────────────────────────
        new_tokens = draft_tokens[:n] + [bonus_token]
        new_tensor = torch.tensor([new_tokens], dtype=torch.long, device=DEVICE)
        current_ids = torch.cat([current_ids, new_tensor], dim=-1)

        # Stop if EOS appears anywhere in the committed tokens
        for tok in new_tokens:
            if tok == eos_token_id:
                generated = current_ids[0, input_ids.shape[1]:]
                return generated, total_draft_tokens, total_accepted_tokens
            if current_ids.shape[1] - input_ids.shape[1] >= max_new_tokens:
                break

    generated = current_ids[0, input_ids.shape[1]:]
    return generated, total_draft_tokens, total_accepted_tokens


# ─── Sanity Check ────────────────────────────────────────────────────────────

def sanity_check(target_model, draft_model, tokenizer, gamma, max_new_tokens):
    prompt   = "write a quick sort algorithm."
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
    print(f"  Gamma (γ)      : {gamma}")
    print(f"  Max new tokens : {max_new_tokens}\n")

    print("Running autoregressive greedy decoding (baseline)…")
    t0          = time.time()
    baseline_ids = autoregressive_greedy_decode(
        target_model, input_ids, max_new_tokens, eos_token_id
    )
    t_baseline  = time.time() - t0
    print(f"  ✔ Done — {len(baseline_ids)} tokens in {t_baseline:.2f}s "
          f"({len(baseline_ids)/t_baseline:.1f} tok/s)")

    print(f"\nRunning speculative decoding (greedy, γ={gamma})…")
    t0 = time.time()
    spec_ids, total_drafted, total_accepted = speculative_decode(
        target_model, draft_model, input_ids, max_new_tokens,
        gamma, eos_token_id, greedy=True,
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
    print(f"  Acceptance rate (α) : {alpha:.4f}  ({total_accepted}/{total_drafted})")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma",          type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    target_model, draft_model, tokenizer = load_models()
    sanity_check(target_model, draft_model, tokenizer,
                 gamma=args.gamma, max_new_tokens=args.max_new_tokens)

if __name__ == "__main__":
    main()