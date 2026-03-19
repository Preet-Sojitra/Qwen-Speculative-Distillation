# Adaptive Speculative Decoding: Improving Acceptance Rate with Teacher-Student Alignment

This repository contains the codebase for our graduate NLP (CS6320) final course project. We implement a custom, from-scratch Speculative Decoding engine in PyTorch and introduce two novel optimizations to accelerate Large Language Model (LLM) inference on domain-specific tasks (Python Code Generation).

## Overview

Speculative Decoding accelerates autoregressive generation by using a smaller draft model to propose tokens, which a larger target model verifies in parallel. However, wall-clock speedups are strictly bounded by the draft model's Acceptance Rate ($\alpha$) and wasted arithmetic operations during periods of high uncertainty. 

Our project improves upon the standard framework \([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)\) through:
1. **Teacher-Student Alignment (Knowledge Distillation):** We align the draft model's probability distribution directly with the target model using soft labels, mitigating stylistic divergence and maximizing $\alpha$.
2. **Dynamic Lookahead Halting:** We replace the static lookahead parameter ($\gamma$) with a dynamic halting mechanism. A lightweight MLP trained on the draft model's token-level entropy halts generation early to prevent wasted FLOPs.

**Models Used:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Target) and `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Draft).

## Repository Structure

The codebase is modularized into four distinct pipelines to support parallel development:

* **`/1_engine`**: Contains the custom PyTorch implementation of the Speculative Decoding algorithm (Algorithm 1), bypassing standard `.generate()` wrappers for raw logit access.
* **`/2_alignment`**: Contains the training scripts for Supervised Fine-Tuning (LoRA) and Knowledge Distillation (KL-Divergence) on the `CodeAlpaca-20k` dataset.
* **`/3_dynamic_halting`**: Contains the architecture and training loop for the 2-layer MLP classifier used to dynamically modulate the lookahead parameter ($\gamma$) based on confidence entropy.
* **`/4_evaluation`**: Contains the benchmarking suite to evaluate Exact Match, Tokens Per Second (TPS), Acceptance Rate ($\alpha$), and FLOP reduction on the `openai_humaneval` dataset.

## 🛠️ Setup & Installation

Ensure you have a GPU environment (CUDA) available. 

```bash
git clone https://github.com/YOUR-USERNAME/Adaptive-SpecDec.git
cd Adaptive-SpecDec
pip install -r requirements.txt