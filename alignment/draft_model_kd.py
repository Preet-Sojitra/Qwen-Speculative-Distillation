"""
Draft Model Knowledge Distillation (KD)
========================================
Distills logit-level knowledge from the target model (Qwen2.5-Coder-7B-Instruct)
into the draft model (Qwen2.5-Coder-0.5B-Instruct) using top-K KL-divergence,
combined with standard cross-entropy on the ground-truth labels.
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: target (7B) → draft (0.5B)"
    )

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="./kd_checkpoints")
    parser.add_argument("--output_dir", type=str, default="./kd_weights")
    parser.add_argument("--save_every_n", type=int, default=500)
    parser.add_argument("--checkpoint_window", type=int, default=3)

    # Models
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--draft_model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)

    # KD hyper-parameters
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for distillation loss (1-alpha = weight for CE loss).")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top-K logits to distill over.")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)

    return parser.parse_args()




def load_models(args: argparse.Namespace):
    """
    Load target (frozen, fp16) and draft (trainable, fp32) models.

    CRITICAL: The draft model MUST be loaded in fp32 (or bf16) for training.
    Loading the *trainable* model in fp16 causes gradient overflow → NaN loss
    after the very first optimizer step, because fp16 has only ~3 decimal digits
    of precision and a tiny dynamic range (max ~65504).
    """
    # --- Target model (frozen, fp16 is fine) ---
    tokenizer_t = AutoTokenizer.from_pretrained(args.target_model)
    model_t = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad = False 

    # --- Draft model (trainable, fp32 for stable training) ---
    tokenizer_d = AutoTokenizer.from_pretrained(args.draft_model)
    model_d = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float32,   # FIX: fp32 to avoid NaN gradients
        device_map="auto",
    )

    return model_t, tokenizer_t, model_d, tokenizer_d


def build_dataloader(tokenizer_d, args: argparse.Namespace) -> DataLoader:
    """Load CodeAlpaca-20k, format as chat, tokenize, and return a DataLoader."""

    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    def format_sample(sample):
        user_content = sample["instruction"]
        if sample["input"].strip():
            user_content += "\n\n" + sample["input"]

        msgs = [
            {"role": "system",    "content": "You are a coding assistant."},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": sample["output"]},
        ]

        formatted = tokenizer_d.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted}

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    def tokenize_sample(sample):
        tokens = tokenizer_d(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )

        labels = tokens["input_ids"].copy()

        # Mask padding tokens so they don't contribute to the CE loss
        pad_id = tokenizer_d.pad_token_id
        labels = [-100 if tok == pad_id else tok for tok in labels]

        tokens["labels"] = labels
        return tokens

    dataset = dataset.map(tokenize_sample, batched=False, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def compute_distillation_loss(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """
    Top-K KL-divergence distillation loss, masked to only count non-padding
    token positions.

    Args:
        target_logits: (B, S, V) float32 logits from the frozen target model.
        draft_logits:  (B, S, V) float32 logits from the trainable draft model.
        attention_mask: (B, S) binary mask (1 = real token, 0 = padding).
        temperature: softmax temperature for softer distributions.
        top_k: number of top vocabulary entries to distill over.

    Returns:
        Scalar KL-divergence loss, averaged over non-padding positions.
    """
    # Select the top-K vocabulary indices according to the teacher
    _, topk_idx = torch.topk(target_logits, k=top_k, dim=-1)       # (B, S, K)
    topk_logits_t = target_logits.gather(-1, topk_idx)              # (B, S, K)
    topk_logits_d = draft_logits.gather(-1, topk_idx)               # (B, S, K)

    # Tempered log-probabilities
    log_p_t = F.log_softmax(topk_logits_t / temperature, dim=-1)    # (B, S, K)
    log_p_d = F.log_softmax(topk_logits_d / temperature, dim=-1)    # (B, S, K)

    # Per-position KL:  sum_k  p_t(k) * [log p_t(k) - log p_d(k)]
    kl_per_pos = (log_p_t.exp() * (log_p_t - log_p_d)).sum(dim=-1) # (B, S)

    # Mask out padding positions
    mask = attention_mask.float()                                    # (B, S)
    kl_per_pos = kl_per_pos * mask

    # Average over non-padding positions (not over the full 512-length)
    num_real_tokens = mask.sum().clamp(min=1.0)
    distill_loss = kl_per_pos.sum() / num_real_tokens

    # Scale by T² (standard distillation gradient correction)
    return distill_loss * (temperature ** 2)


def save_checkpoint(model_d, tokenizer_d, optimizer, epoch, step, loss, args):
    ckpt_name = f"epoch{epoch + 1}_step{step + 1}"
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
    os.makedirs(ckpt_path, exist_ok=True)

    model_d.save_pretrained(ckpt_path)
    tokenizer_d.save_pretrained(ckpt_path)
    torch.save({
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, os.path.join(ckpt_path, "training_state.pt"))


    all_ckpts = sorted([
        d for d in os.listdir(args.checkpoint_dir)
        if os.path.isdir(os.path.join(args.checkpoint_dir, d))
    ])
    while len(all_ckpts) > args.checkpoint_window:
        old = all_ckpts.pop(0)
        shutil.rmtree(os.path.join(args.checkpoint_dir, old))

    print(f"  💾 Checkpoint saved: {ckpt_name}")


def train(args: argparse.Namespace):
    assert torch.cuda.is_available(), "ERROR: CUDA GPU required."
    device = "cuda"


    model_t, tokenizer_t, model_d, tokenizer_d = load_models(args)

    dataloader = build_dataloader(tokenizer_d, args)

    optimizer = AdamW(model_d.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model_t.eval()

    for epoch in range(args.epochs):
        model_d.train()

        total_loss = 0.0
        total_dist_loss = 0.0
        total_ce_loss = 0.0
        num_valid_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # ── Teacher forward (no grad, fp16) ──
            with torch.no_grad():
                output_t = model_t(input_ids=input_ids, attention_mask=attention_mask)
                target_logits = output_t.logits.float()  # upcast to fp32 for softmax

            # ── Student forward (grad, fp32) ──
            output_d = model_d(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            draft_logits = output_d.logits       # already fp32
            ce_loss = output_d.loss              # cross-entropy on ground truth

            # ── Distillation loss ──
            distill_loss = compute_distillation_loss(
                target_logits=target_logits,
                draft_logits=draft_logits,
                attention_mask=attention_mask,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # ── Combined loss ──
            loss = args.alpha * distill_loss + (1 - args.alpha) * ce_loss

            # Guard against NaN / exploding loss
            if loss.isnan() or loss.isinf():
                print(f"  ⚠️  Skipping step {step} — loss={loss.item()}")
                optimizer.zero_grad()
                continue

            # ── Backward + update ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # ── Tracking ──
            total_loss += loss.item()
            total_dist_loss += distill_loss.item()
            total_ce_loss += ce_loss.item()
            num_valid_steps += 1

            # ── Logging ──
            if (step + 1) % args.log_every == 0 or step == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{step + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Distill: {distill_loss.item():.4f} | "
                    f"CE: {ce_loss.item():.4f}"
                )

            # ── Checkpoint ──
            if (step + 1) % args.save_every_n == 0:
                save_checkpoint(
                    model_d, tokenizer_d, optimizer,
                    epoch, step, loss.item(), args
                )

        # ── Epoch summary ──
        n = max(num_valid_steps, 1)
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1} Complete")
        print(f"  Avg Total Loss  : {total_loss / n:.4f}")
        print(f"  Avg Distill Loss: {total_dist_loss / n:.4f}")
        print(f"  Avg CE Loss     : {total_ce_loss / n:.4f}")
        print(f"  Valid steps     : {num_valid_steps}/{len(dataloader)}")
        print(f"{'=' * 60}\n")

    model_d.save_pretrained(args.output_dir)
    tokenizer_d.save_pretrained(args.output_dir)
    torch.save({
        "epoch": args.epochs - 1,
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(args.output_dir, "training_state.pt"))
    print(f"\n✅ Final weights saved to: {args.output_dir}")

if __name__ == "__main__":
    train(parse_args())
