import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "..")
from config import *

def load_models(draft_model_path=None):
    """Load target + draft models and shared tokenizer."""
    print(f"[INFO] Loading target model: {TARGET_MODEL_ID}")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    ).eval()

    if draft_model_path is not None:
        print(f"[INFO] Loading KD draft model from: {draft_model_path}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_ID, dtype=DTYPE, device_map=DEVICE,
        ).eval()
        state_dict = torch.load(draft_model_path, map_location=DEVICE)
        draft_model.load_state_dict(state_dict, strict=False)
        print("[INFO] KD draft weights loaded successfully.")
    else:
        print(f"[INFO] Loading baseline draft model: {DRAFT_MODEL_ID}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_ID, dtype=DTYPE, device_map=DEVICE,
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    return target_model, draft_model, tokenizer