import torch

TARGET_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATASET_ID      = "sahil2801/CodeAlpaca-20k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32