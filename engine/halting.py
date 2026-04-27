"""
engine/halting.py
─────────────────
Utility to load the trained Dynamic-Halting MLP and expose a lightweight
predict function for use inside speculative_dynamic().

The MLP takes z-score-normalised (entropy, max_prob) and outputs a scalar
in [0, 1] representing predicted acceptance probability.
"""

import json
import torch
from pathlib import Path
from dynamic_halting.model import DynamicHaltingMLP

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "mlp_weights.pt"
_DEFAULT_NORM    = Path(__file__).resolve().parent.parent / "weights" / "norm_params.json"


def load_halting_mlp(
    weights_path: str | Path | None = None,
    norm_params_path: str | Path | None = None,
    device: str = "cpu",
):
    """
    Load the halting MLP and normalisation statistics.

    Returns
    -------
    predict_fn : callable(entropy: float, max_prob: float) → float
        Returns the predicted acceptance probability in [0, 1].
    """
    weights_path     = Path(weights_path)     if weights_path     else _DEFAULT_WEIGHTS
    norm_params_path = Path(norm_params_path) if norm_params_path else _DEFAULT_NORM

    with open(norm_params_path, "r") as f:
        norm = json.load(f)
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    std  = torch.tensor(norm["std"],  dtype=torch.float32, device=device)

    model = DynamicHaltingMLP(input_dim=2, hidden_dim=16)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    @torch.no_grad()
    def predict_fn(entropy: float, max_prob: float) -> float:
        """Return predicted P(accept) for a single draft token."""
        raw = torch.tensor([[entropy, max_prob]], dtype=torch.float32, device=device)
        normed = (raw - mean) / std
        return model(normed).item()

    return predict_fn
