import sys

import clip
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import src.model_utils
from PIL import Image

from src.prepare_training_set import get_headers
import src.utils
model = None
device = None
preprocess = None
def init():
    global model, device, preprocess
    device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_base_model():
    global model, device, preprocess
# ──  Load Model ─────────────────────────────────────────────
    print(f"Using device: {device}")
    model, preprocess = src.model_utils.load_model("ViT-B/32")
    model.eval()

def load_finetuned_model():
    global model, device, preprocess
# ──  Load Model ─────────────────────────────────────────────
    print(f"Using device: {device}")
    model, preprocess = src.model_utils.load_model("ViT-B/32")
    checkpoint = torch.load("output/clip_finetuned.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

def get_probabilities(row):
    image = preprocess(Image.open(row["local_path"])).unsqueeze(0).to(device)

    headers = get_headers("datasets/csvs/training/normalized_output.csv")
    available_instruments = src.utils.get_instrument_labels(headers)
    label = row["instrument"]
    texts = clip.tokenize(available_instruments).to(device)
    idx = available_instruments.index(label)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # for text, prob in zip(available_instruments, similarity[0]):
    #     print(f"{label} {text}: {prob:.4f}")
    # #print(f"Probability of {label} is {similarity[0][idx]}")
    # #print(similarity[0][idx].item())
    #print(f"---------------RETURNING {similarity[0][idx].item()}")
    get_highest_score(row["local_path"], available_instruments, similarity)
    return similarity[0][idx].item()

def check_base_model():
    load_base_model()

def check_finetuned_model():
    load_finetuned_model()

def get_highest_score(local_file, available_instruments, similarity):
    scores = {text: prob.item() for text, prob in zip(available_instruments, similarity[0])}
    best = max(scores, key=scores.get)
    print(f"local_file = {local_file} Best match: {best} → {scores[best]:.4f}")

def run_untrained_inference_loop():
    inference_df = pd.read_csv("datasets/csvs/inference/inference_objects.csv")
    inference_df["untrained_guess"] = None
    for index, row in inference_df.iterrows():
        probability = get_probabilities(row)
        inference_df.at[index, "untrained_guess"] = probability
    src.utils.export_to_csv(inference_df, "datasets/csvs/inference/probabilities.csv")

def run_trained_inference_loop():
    inference_df = pd.read_csv("datasets/csvs/inference/probabilities.csv")
    inference_df["trained_guess"] = None
    for index, row in inference_df.iterrows():
        probability = get_probabilities(row)
        inference_df.at[index, "trained_guess"] = probability

    src.utils.export_to_csv(inference_df, "datasets/csvs/inference/probabilities.csv")



if __name__ == "__main__":
    init()
    if sys.argv[1] == "base":
        load_base_model()
        run_untrained_inference_loop()
    elif sys.argv[1] == "trained":
        load_finetuned_model()
        run_trained_inference_loop()
