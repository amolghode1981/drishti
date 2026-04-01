import clip
import torch

def load_model(modelString):
# ──  Load Model ─────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load(modelString, device=device)
    return model, preprocess