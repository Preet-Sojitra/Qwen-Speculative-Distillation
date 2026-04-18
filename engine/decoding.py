import torch
import torch.nn.functional as F

@torch.no_grad()
def autoregressive(model, input_ids, max_new_tokens, eos_token_id):
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        logits    = model(generated_ids).logits          # (1, seq, vocab)
        next_tok  = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
        generated_ids = torch.cat([generated_ids, next_tok], dim=-1)
        if next_tok.item() == eos_token_id:
            break

    # Return only the newly generated part (strip the prompt)
    return generated_ids[0, input_ids.shape[1]:]

@torch.no_grad()
def speculative(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens,
    gamma,
    eos_token_id,
    device,
    greedy=True,
    log_features=False,
):
    """
    Speculative Decoding — Algorithm 1 from Leviathan et al. (2023).
    No KV-cache: every forward pass reprocesses the full sequence.

    Args:
        log_features: If True, compute per-draft-token features (entropy,
                      max_prob, accepted) for MLP training. A 4th element
                      `token_records` is appended to the return tuple.

    Returns:
        generated_ids         — 1-D tensor of new token IDs (no prompt)
        total_draft_tokens    — total tokens proposed by the draft model
        total_accepted_tokens — draft tokens accepted by the target model
        token_records         — (only when log_features=True) list[dict] with
                                keys {entropy, max_prob, accepted}
    """
    # current_ids holds the full sequence: prompt + all committed tokens so far
    current_ids = input_ids.clone()

    total_draft_tokens    = 0
    total_accepted_tokens = 0
    token_records         = [] if log_features else None

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
        draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
        verify_ids   = torch.cat([current_ids, draft_tensor], dim=-1)

        p_logits = target_model(verify_ids).logits   # (1, prefix+γ, vocab)
        prefix_len = current_ids.shape[1]            # length before drafts

        # ── STEP 3: Accept / Reject ───────────────────────────────────────────
        n = 0  # number of accepted draft tokens
        if greedy:
            for j in range(current_gamma):
                # logit at position (prefix_len - 1 + j) gives p(· | ctx + draft[:j])
                p_j = F.softmax(p_logits[:, prefix_len - 1 + j, :], dim=-1)
                accepted = int(torch.argmax(p_j, dim=-1).item() == draft_tokens[j])

                if log_features:
                    q_j      = draft_probs[j]
                    q_j_f    = q_j.float()
                    entropy  = -(q_j_f * torch.log(q_j_f + 1e-10)).sum().item()
                    max_prob = q_j.max().item()
                    token_records.append({
                        "entropy":  entropy,
                        "max_prob": max_prob,
                        "accepted": accepted,
                    })

                if accepted:
                    n += 1
                else:
                    break
        else:
            for j in range(current_gamma):
                p_j = F.softmax(p_logits[:, prefix_len - 1 + j, :], dim=-1)
                q_j = draft_probs[j]
                x_j = draft_tokens[j]
                r   = torch.rand(1, device=device)
                accepted = int(
                    r.item() < min(1.0, (p_j[0, x_j] / q_j[0, x_j]).item())
                )

                if log_features:
                    q_j_f    = q_j.float()
                    entropy  = -(q_j_f * torch.log(q_j_f + 1e-10)).sum().item()
                    max_prob = q_j.max().item()
                    token_records.append({
                        "entropy":  entropy,
                        "max_prob": max_prob,
                        "accepted": accepted,
                    })

                if accepted:
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

        # Check for EOS *before* committing, truncate new_tokens at EOS
        eos_pos = None
        for i, tok in enumerate(new_tokens):
            if tok == eos_token_id:
                eos_pos = i
                break

        if eos_pos is not None:
            new_tokens = new_tokens[:eos_pos + 1]   # keep EOS, drop anything after

        new_tensor = torch.tensor([new_tokens], dtype=torch.long, device=device)
        current_ids = torch.cat([current_ids, new_tensor], dim=-1)

        # Now check length cap too
        tokens_generated = current_ids.shape[1] - input_ids.shape[1]
        if eos_pos is not None or tokens_generated >= max_new_tokens:
            generated = current_ids[0, input_ids.shape[1]:]
            # Truncate to exactly max_new_tokens if we overshot
            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]

            if log_features:
                return generated, total_draft_tokens, total_accepted_tokens, token_records
            return generated, total_draft_tokens, total_accepted_tokens
