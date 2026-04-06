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


def print_matches(similarity, labels):
    percentages = (similarity[0] * 100).round(decimals=2)
    rows = []
    for label, score in zip(labels, percentages):
        data = {
            "label": label,
            "score": f"{score:.2f}%"
        }
        rows.append(data);
    print(rows)
    import json
    return json.dumps(rows, indent=2)

def get_binary_probabilities(row):
    image = preprocess(Image.open(row["local_path"])).unsqueeze(0).to(device)
    #texts = clip.tokenize(available_instruments).to(device)
    # texts = clip.tokenize([
    #         "An image of an Indian miniature painting depicting music making activity. ",
    #         "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
    # ]).to(device)
    labels = [
        "a person playing a musical instrument",
        "a musician playing an instrument",
        "a person holding a string instrument",
        "a person playing barrel drum",
        "a portrait of a person",
        "a person standing without objects",
        "a traditional painting of people",
        "a drawing of a person without instruments"
    ]
    texts = clip.tokenize(labels).to(device)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    #result = f"Has music : {similarity[0][0].item():.4f} No music : {similarity[0][1].item():.4f} ";
    return print_matches(similarity, labels)

#improves labels and the temperature to ampify the difference.
def get_binary_probabilities_improved(row):
    image = preprocess(Image.open(row["local_path"])).unsqueeze(0).to(device)
    #texts = clip.tokenize(available_instruments).to(device)
    # texts = clip.tokenize([
    #         "An image of an Indian miniature painting depicting music making activity. ",
    #         "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
    # ]).to(device)
    # labels = [
    #     "a person playing a musical instrument",
    #     "a musician playing an instrument",
    #     "a person holding a string instrument",
    #     "a person playing barrel drum",
    #     "a portrait of a person",
    #     "a person standing without objects",
    #     "a traditional painting of people",
    #     "a drawing of a person without instruments"
    # ]
    labels = [
        "a person playing a musical instrument",
        "a person playing a barrel drum",
        "a person playing a gourd instrument",
        "an empty room with no people",
        "an aerial view of a landscape",       # very different visually
        "a close up of food on a plate",       # no person at all
    ]
    texts = clip.tokenize(labels).to(device)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Similarity
        temperature = 0.07
        '''
        We found the image embeddings and text embeddings. Now multiply that
        and ampify the difference.
        '''
        logits = image_features @ text_features.T

        similarity = (logits/temperature).softmax(dim=-1)

    #result = f"Has music : {similarity[0][0].item():.4f} No music : {similarity[0][1].item():.4f} ";
    return print_matches(similarity, labels)

def test_single_image():
    image = preprocess(Image.open("data/inference/random_images/2015-05-28-1864.jpeg")).unsqueeze(0).to(device)
    #texts = clip.tokenize(available_instruments).to(device)
    # texts = clip.tokenize([
    #         "An image of an Indian miniature painting depicting music making activity. ",
    #         "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
    # ]).to(device)
    # labels = [
    #     "a person playing a musical instrument",
    #     "a musician playing an instrument",
    #     "a person holding a violin instrument",
    #     "a person playing barrel drum",
    #     "a portrait of a person",
    #     "a person standing without objects",
    #     "a traditional painting of people",
    #     "a drawing of a person without instruments"
    # ]
    # labels = [
    #     "a person playing violin",           # specific positive
    #     "a person playing a barrel drum",    # specific positive
    #     "a person playing a gourd instrument", # specific positive
    #     "an empty room with no people",      # strong negative
    #     "a person cooking food",             # plausible but wrong
    #     "a landscape photograph",            # strong negative
    # ]
    labels = [
        "a person playing violin",
        "a person playing a barrel drum",
        "a person playing a gourd instrument",
        "an empty room with no people",
        "an aerial view of a landscape",       # very different visually
        "a close up of food on a plate",       # no person at all
    ]
    texts = clip.tokenize(labels).to(device)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features[0, :10])
        text_features = model.encode_text(texts)
        print(text_features[0, :10])
        # Normalize i.e converting the vector into unit vector
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print(image_features[0, :10])
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(text_features[0, :10])
        # Similarity
        # Sharpened (divide by temperature before softmax)
        temperature = 0.07
        '''
        We found the image embeddings and text embeddings. Now multiply that
        and ampify the difference.
        '''
        logits = image_features @ text_features.T
        print(list(zip(labels, logits[0].tolist())))
        # Softmax different from normalization.
        '''
        softmax(z_i) = e^z_i / sum(e^z_j)  for j = 1 to n
        '''
        similarity = (logits / temperature).softmax(dim=-1)

    #result = f"Has music : {similarity[0][0].item():.4f} No music : {similarity[0][1].item():.4f} ";
    return print_matches(similarity, labels)


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
        result = get_binary_probabilities_improved(row)
        #result = test_single_image()
        inference_df.at[index, "untrained_guess"] = result
        if index > 4:
            break
    src.utils.export_to_csv(inference_df, "output/probabilities.csv")

def run_binary_trained_inference_loop():
    inference_df = pd.read_csv("output/probabilities.csv")
    inference_df["trained_guess"] = None
    for index, row in inference_df.iterrows():
        result = get_binary_probabilities_improved(row)
        #result = test_single_image()
        inference_df.at[index, "trained_guess"] = result
        if index > 4:
            break

    src.utils.export_to_csv(inference_df, "output/probabilities.csv")


def run_binary_untrained_inference_single_prediction():
    inference_df = pd.read_csv("datasets/csvs/inference/inference_objects.csv")
    inference_df["untrained_guess"] = None
    test_single_image()
    # inference_df.at[index, "untrained_guess"] = result

def run_binary_trained_inference_single_prediction():
    inference_df = pd.read_csv("output/probabilities.csv")
    inference_df["trained_guess"] = None
    test_single_image()

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
    #run_binary_untrained_inference_single_prediction()
    run_binary_untrained_inference_loop()
    cleanup()
    load_finetuned_model()
    # run_binary_trained_inference_single_prediction()
    run_binary_trained_inference_loop()

if __name__ == "__main__":
    init()
    run_binary_inference_loop()