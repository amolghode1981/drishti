import gc
import sys

import clip
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import src.model_utils
from PIL import Image

from src.utils import get_headers
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


def get_binary_probabilities(row):
    image = preprocess(Image.open(row["local_path"])).unsqueeze(0).to(device)
    #texts = clip.tokenize(available_instruments).to(device)
    texts = clip.tokenize([
            "An image of an Indian miniature painting depicting music making activity. ",
            "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
    ]).to(device)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    result = f"Has music : {similarity[0][0].item():.4f} No music : {similarity[0][1].item():.4f} ";
    return result


def check_base_model():
    load_base_model()

def check_finetuned_model():
    load_finetuned_model()

# def get_highest_score(local_file, available_instruments, similarity):
#     scores = {text: prob.item() for text, prob in zip(available_instruments, similarity[0])}
#     best = max(scores, key=scores.get)
#     print(f"local_file = {local_file} Best match: {best} → {scores[best]:.4f}")


def run_binary_untrained_inference_loop():
    inference_df = pd.read_csv("datasets/csvs/inference/inference_objects.csv")
    inference_df["untrained_guess"] = None
    for index, row in inference_df.iterrows():
        result = get_binary_probabilities(row)
        inference_df.at[index, "untrained_guess"] = result
    src.utils.export_to_csv(inference_df, "output/probabilities.csv")

def run_binary_trained_inference_loop():
    inference_df = pd.read_csv("output/probabilities.csv")
    inference_df["trained_guess"] = None
    for index, row in inference_df.iterrows():
        result = get_binary_probabilities(row)
        inference_df.at[index, "trained_guess"] = result
    src.utils.export_to_csv(inference_df, "output/probabilities.csv")


def cleanup():

    global model, preprocess
    del model
    del preprocess
    gc.collect()  # force Python garbage collection

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_binary_inference_loop():
    init()
    load_base_model()
    run_binary_untrained_inference_loop()
    cleanup()
    load_finetuned_model()
    run_binary_trained_inference_loop()

if __name__ == "__main__":
    init()
    run_binary_inference_loop()