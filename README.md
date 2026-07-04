# Adaptive Speculative Decoding: Improving Acceptance Rate with Teacher-Student Alignment

> **CS 6320 — Graduate NLP Final Project**

[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/F3jK3KYNLcY)
[![Model Weights](https://img.shields.io/badge/Model%20Weights-Hugging%20Face-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/preetsojitra/Qwen-Speculative-Distill/tree/main)
[![Project Report](https://img.shields.io/badge/Project%20Report-PDF-blueviolet?style=for-the-badge)](/Final-Report.pdf)

This repository implements a custom, from-scratch Speculative Decoding engine in PyTorch to accelerate Large Language Model (LLM) inference on Python code generation. It introduces two primary optimizations to the standard speculative decoding framework:

1. **Teacher-Student Alignment (Knowledge Distillation):** Aligns the draft model's probability distribution directly with the target model using top-K KL-divergence loss to maximize acceptance rate ($\alpha$). See the [Final-Report.pdf](/Final-Report.pdf) for the comprehensive theoretical analysis, methodology, and detailed findings.
2. **Dynamic Lookahead Halting:** Replaces static speculation lookahead ($\gamma$) with a dynamic halting mechanism. A lightweight 2-layer MLP trained on draft model token-level entropy and maximum probability halts speculation early when uncertainty is high, preventing wasted FLOPs.

---

## 📊 Evaluation Results

### Table 1: Alignment Ablation
*Compares Baseline vs. SFT vs. KD draft models on OpenAI HumanEval (evaluating acceptance rate $\alpha$, tokens per step, and wall-clock speedup).*

| Draft Model Type | Acceptance Rate ($\alpha$) | Tokens/Step | Wall-Clock Speedup  | Final TPS |
|---|---|---|---|---|
| Off-the-shelf 0.5B (Baseline) | 28.4% | 1.39 | 1.05x | 26.2 tok/s |
| LoRA SFT 0.5B | 55.2% | 2.11 | 1.60x | 40.0 tok/s |
| **Knowledge Distillation (KD) 0.5B** | **76.5%** | **3.14** | **2.38x** | **59.5 tok/s** |

### Table 2: Dynamic Lookahead
*Compares Fixed $\gamma$ vs. Dynamic $\gamma$ speculative decoding (evaluating draft forward passes, wasted tokens, and throughput).*

| Lookahead Strategy | Draft FWD Passes | Wasted Tokens / Step | Final TPS  | Speedup (vs AR) |
|---|---|---|---|---|
| KD 0.5B (Fixed $\gamma=4$) | 6368 | 2962 | 59.5 tok/s | 2.38x |
| **KD 0.5B + Dynamic $\gamma$ (MLP Halting)** | **3683** | **439** | **64.5 tok/s** | **2.58x** |

---

## 💻 Hardware Requirements

| Task | GPU Required? | Minimum VRAM | Note |
|---|---|---|---|
| **Gradio Demo** | ❌ No | — | Runs on CPU using pre-recorded race data. |
| **MLP Training** | ❌ No | — | Trains on CPU in seconds. |
| **Distillation (KD) & Evaluation** | ✅ Yes | ~24 GB | Loads 7B and 0.5B models simultaneously. |

---

## 🚀 Quick Start (Gradio Demo)

Run the interactive Gradio demo locally to visualize standard autoregressive decoding vs. speculative decoding side-by-side:

```bash
pip install transformers gradio
python [demo/app.py](/demo/app.py)
```

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/Preet-Sojitra/Qwen-Speculative-Distillation.git
cd Qwen-Speculative-Distillation
python -m venv venv && source venv/bin/activate

# Install PyTorch and core dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Adjust for CUDA version
pip install transformers datasets accelerate peft trl gradio tqdm pandas
```
*(Optional SFT acceleration: `pip install unsloth`)*

### Download Pre-trained Weights
Download trained draft and MLP weights from the [Hugging Face Repository](https://huggingface.co/preetsojitra/Qwen-Speculative-Distill/tree/main) and place them in the `weights/` directory:
- `weights/kd_model/` (KD-distilled draft checkpoints)
- `weights/mlp/` (trained halting MLP weights and normalization parameters)
- `weights/sft_model.zip` (LoRA SFT adapter weights)

---

## 📊 Core Pipelines

All commands should be run from the repository root.

### 1. Speculative Decoding Engine
The speculative engine in [engine/decoding.py](/engine/decoding.py) provides:
- `autoregressive()`: Greedy autoregressive decoding baseline.
- `speculative()`: Fixed-$\gamma$ speculation ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)).
- `speculative_dynamic()`: Dynamic-$\gamma$ speculation using the halting MLP.

Run the test suite to verify output exact-match correctness:
```bash
python -m tests.test_baseline
```

### 2. Draft Model Alignment
Train the draft model (`Qwen2.5-Coder-0.5B-Instruct`) to match the target (`Qwen2.5-Coder-7B-Instruct`) on `sahil2801/CodeAlpaca-20k`:

- **Option A: Supervised Fine-Tuning (SFT with LoRA)**
  ```bash
  python -m alignment.draft_model_sft --output_dir ./weights/sft_model
  ```
- **Option B: Knowledge Distillation (KD)**
  ```bash
  python -m alignment.draft_model_kd --output_dir ./weights/kd_model --alpha 0.7 --top_k 50
  ```

### 3. Dynamic Halting MLP
1. **Generate training features** (entropy, max probability, and acceptance target):
   ```bash
   python -m dynamic_halting.generate_csv --output data/data_for_MLP.csv
   ```
   *(A pre-generated CSV is already included in `data/data_for_MLP.csv`)*
2. **Train the halting classifier**:
   ```bash
   cd dynamic_halting && python train.py
   ```

### 4. Benchmark Evaluation
Compare alignment models and dynamic lookahead on OpenAI HumanEval:
```bash
python -m evaluation.evaluator --kd_draft_path ./weights/kd_model --sft_draft_path ./weights/sft_model
```

---

## 👥 Acknowledgments

- [Leviathan et al., 2023 — *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)
- [Qwen2.5-Coder](https://huggingface.co/Qwen) model family by Alibaba
- [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) dataset
- [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning