# Adaptive Speculative Decoding: Improving Acceptance Rate with Teacher-Student Alignment

> **CS 6320 — Graduate NLP Final Project**

This repository implements a custom, from-scratch Speculative Decoding engine in PyTorch and introduces two novel optimizations to accelerate Large Language Model (LLM) inference on domain-specific tasks (Python Code Generation).

---

## Overview

Speculative Decoding accelerates autoregressive generation by using a smaller **draft model** to propose tokens, which a larger **target model** verifies in parallel. However, wall-clock speedups are strictly bounded by the draft model's **Acceptance Rate (α)** and wasted arithmetic operations during periods of high uncertainty.

Our project improves upon the standard framework ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) through:

1. **Teacher-Student Alignment (Knowledge Distillation):** We align the draft model's probability distribution directly with the target model using soft labels (top-K KL-divergence), mitigating stylistic divergence and maximizing α.
2. **Dynamic Lookahead Halting:** We replace the static lookahead parameter (γ) with a dynamic halting mechanism. A lightweight 2-layer MLP trained on the draft model's token-level entropy halts generation early to prevent wasted FLOPs.

**Models Used:**

| Role | Model | Size |
|------|-------|------|
| Target (Verifier) | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B params |
| Draft (Proposer) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 0.5B params |

**Dataset:** [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) (SFT & KD training), [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) (evaluation)

---

## Repository Structure

```
.
├── engine/                     # Core speculative decoding algorithms
│   ├── decoding.py             #   autoregressive(), speculative(), speculative_dynamic()
│   └── halting.py              #   Load trained MLP for dynamic γ halting
│
├── alignment/                  # Draft model training pipelines
│   ├── draft_model_sft.py      #   LoRA SFT via Unsloth + TRL
│   └── draft_model_kd.py       #   Full-weight KD (top-K KL + CE)
│
├── dynamic_halting/            # Dynamic lookahead MLP
│   ├── generate_csv.py         #   Generate (entropy, max_prob, accepted) features
│   ├── dataset.py              #   PyTorch Dataset with z-score normalization
│   ├── model.py                #   2-layer MLP architecture (2 → 16 → 1)
│   └── train.py                #   Training loop with early stopping
│
├── evaluation/                 # Benchmarking suite
│   └── evaluator.py            #   Produces Table 1 (Alignment) & Table 2 (Dynamic γ)
│
├── demo/                       # Gradio demo (CPU-friendly)
│   ├── app.py                  #   Interactive inference race visualization
│   └── demo_race_data.json     #   Pre-recorded race data (no GPU needed)
│
├── tests/                      # Sanity checks & integration tests
│   ├── test_baseline.py        #   Exact-match: autoregressive vs speculative
│   ├── test_sft_inference.py   #   SFT draft model inference test
│   ├── test_kd_inference.py    #   KD draft model inference test
│   └── test_dynamic_halting.py #   Dynamic γ end-to-end test
│
├── data/                       # Generated datasets
│   └── data_for_MLP.csv        #   Pre-generated MLP training features
│
├── weights/                    # Trained model weights (git-ignored, see below)
│   ├── kd_model/               #   KD draft model checkpoints (.zip archives)
│   ├── mlp/                    #   Halting MLP weights + normalization params
│   ├── sft_model.zip           #   SFT LoRA adapter weights
│   └── sft_weight_onbase.zip   #   SFT weights on base model variant
│
├── config.py                   # Global model IDs, device, and dtype config
├── utils/load_model.py         # Shared model loading utility
└── README.md
```

---

## ⚠️ Hardware Requirements

> [!CAUTION]
> **GPU Access is Required for Training and Evaluation.**
> The training pipelines (SFT, KD) and the full evaluation suite load both the 7B target model and the 0.5B draft model into VRAM simultaneously. This requires a **CUDA-capable GPU with ≥ 24 GB VRAM** (e.g., NVIDIA A100, RTX 4090, or equivalent).

| Task | GPU Required? | Minimum VRAM | Notes |
|------|:---:|:---:|-------|
| **Gradio Demo** (`demo/app.py`) | ❌ No | — | Runs on **CPU only**. Uses pre-recorded data. |
| **MLP Training** (`dynamic_halting/train.py`) | ❌ No | — | Tiny 2-layer MLP; trains on CPU in seconds. |
| **MLP Data Generation** (`dynamic_halting/generate_csv.py`) | ✅ Yes | ~20 GB | Runs speculative decoding to log features. |
| **SFT Training** (`alignment/draft_model_sft.py`) | ✅ Yes | ~16 GB | Uses 4-bit QLoRA via Unsloth. |
| **KD Training** (`alignment/draft_model_kd.py`) | ✅ Yes | ~24 GB | Loads 7B (fp16, frozen) + 0.5B (fp32, trainable). |
| **Full Evaluation** (`evaluation/evaluator.py`) | ✅ Yes | ~20 GB | Loads 7B + 0.5B for end-to-end benchmarks. |
| **Tests** (`tests/`) | ✅ Yes | ~20 GB | All tests run live inference on GPU. |

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Preet-Sojitra/Qwen-Speculative-Distillation.git
cd Qwen-Speculative-Distillation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # adjust for your CUDA version
pip install transformers datasets accelerate peft trl gradio tqdm pandas
```

**For SFT training only** (uses Unsloth for efficient LoRA):
```bash
pip install unsloth
```

> [!NOTE]
> If you only want to run the **Gradio demo**, you just need:
> ```bash
> pip install transformers gradio
> ```

---

## Quick Start — Gradio Demo (No GPU Needed)

The fastest way to see the project in action. The demo replays a **pre-recorded inference race** between standard Autoregressive decoding and our Speculative Decoding approach — no GPU or model downloads required.

```bash
cd demo
python app.py
```

This launches a local Gradio web app. Click **"🏁 Start Inference Comparison"** to watch the side-by-side token generation race with live elapsed-time counters. The speculative decoder finishes significantly faster, visually demonstrating the speedup.

> [!TIP]
> The race data was pre-generated on GPU and saved to `demo_race_data.json`. The demo simply replays token timestamps, so it runs entirely on CPU and requires only the tokenizer download (~100 MB).

---

## Full Pipeline — Step-by-Step (GPU Required)

The following sections walk through each pipeline stage in order. All commands are run from the **project root directory**.

### Pipeline 1: Speculative Decoding Engine

The core algorithm lives in `engine/decoding.py` and provides three functions:

| Function | Description |
|----------|-------------|
| `autoregressive()` | Standard greedy autoregressive decoding (baseline). |
| `speculative()` | Algorithm 1 from Leviathan et al. — fixed γ speculation with verify-then-accept. |
| `speculative_dynamic()` | Our extension — dynamic γ with MLP-based halting. |

**Sanity Check** (verifies speculative output matches autoregressive exactly):
```bash
python -m tests.test_baseline
```

---

### Pipeline 2: Draft Model Alignment

We train the draft model via two approaches to improve its acceptance rate (α).

#### Option A: Supervised Fine-Tuning (SFT with LoRA)

Uses [Unsloth](https://github.com/unslothai/unsloth) for efficient 4-bit QLoRA fine-tuning on CodeAlpaca-20k.

```bash
python -m alignment.draft_model_sft \
    --output_dir ./weights/sft_model \
    --checkpoint_dir ./sft_checkpoints \
    --num_train_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4
```

**Test SFT inference:**
```bash
python -m tests.test_sft_inference
```

#### Option B: Knowledge Distillation (KD)

Distills soft logits from the 7B target into the 0.5B draft using top-K KL-divergence loss combined with cross-entropy.

```bash
python -m alignment.draft_model_kd \
    --output_dir ./weights/kd_model \
    --checkpoint_dir ./kd_checkpoints \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-5 \
    --temperature 2.0 \
    --alpha 0.7 \
    --top_k 50
```

Key hyperparameters:
- `--temperature`: Softmax temperature for KL divergence (default: 2.0)
- `--alpha`: Weight for distillation loss vs CE loss (default: 0.7 = 70% KD + 30% CE)
- `--top_k`: Number of top logits to distill over (default: 50)

**Test KD inference:**
```bash
python -m tests.test_kd_inference
```

---

### Pipeline 3: Dynamic Halting MLP

#### Step 3a: Generate Training Data (GPU Required)

Runs speculative decoding with `log_features=True` to collect per-token `(entropy, max_prob, accepted)` triples:

```bash
python -m dynamic_halting.generate_csv \
    --num_prompts 1000 \
    --gamma 4 \
    --max_new_tokens 128 \
    --output data/data_for_MLP.csv
```

> [!NOTE]
> We have already included a pre-generated `data/data_for_MLP.csv` in the repository (generated from 1,000+ prompts on GPU), so you can **skip this step** and proceed directly to training.

#### Step 3b: Train the Halting MLP (CPU OK)

Trains a small `(2 → 16 → 1)` MLP with binary cross-entropy, early stopping, and best-weight checkpointing:

```bash
cd dynamic_halting
python train.py
```

The best weights are saved to `weights/mlp/mlp_weights.pt` and normalization parameters to `weights/mlp/norm_params.json`.

**Test dynamic halting end-to-end:**
```bash
python -m tests.test_dynamic_halting
```

---

### Pipeline 4: Evaluation

The evaluator produces two result tables benchmarked on [OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval):

- **Table 1 — Alignment Ablation:** Compares Baseline vs SFT vs KD draft models (α, Tokens/Step, Wall-Clock Speedup)
- **Table 2 — Dynamic Lookahead:** Compares Fixed γ vs Dynamic γ (Draft FWD Passes, Wasted Tokens, TPS)

```bash
python -m evaluation.evaluator \
    --num_prompts 20 \
    --max_new_tokens 512 \
    --gamma 4 \
    --kd_draft_path ./weights/kd_model \
    --sft_draft_path ./weights/sft_model \
    --max_gamma 6 \
    --min_gamma 1 \
    --halt_threshold 0.5
```

Results are saved to `evaluation/results/evaluation_results.json`.

---

## Pre-trained Weights

The `weights/` directory contains our trained checkpoints (git-ignored due to size). The key files are:

| File | Description |
|------|-------------|
| `weights/kd_model/` | Full KD-distilled draft model (multiple checkpoint archives) |
| `weights/mlp/mlp_weights.pt` | Trained halting MLP weights (~3 KB) |
| `weights/mlp/norm_params.json` | Z-score normalization stats for MLP input |
| `weights/sft_model.zip` | LoRA SFT adapter weights |

> [!IMPORTANT]
> If weights are not included in the cloned repository, you will need to either:
> 1. Train them yourself using the pipelines above (GPU required), or
> 2. Download them from the provided link (if available from the authors).

---

## Configuration

Global settings are defined in `config.py`:

```python
TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATASET_ID      = "sahil2801/CodeAlpaca-20k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
```

The config auto-detects CUDA availability and adjusts precision accordingly.

---

## Acknowledgments

- [Leviathan et al., 2023 — *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)
- [Qwen2.5-Coder](https://huggingface.co/Qwen) model family by Alibaba
- [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) dataset
- [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning