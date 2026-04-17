import gc
import sys

import clip
import pandas as pd
import torch
import src.model_utils
from PIL import Image
from pathlib import Path
import argparse
import src.web.labels

model = None
device = None
preprocess = None
'''
This is a clean code picked up from bulk_met_zeroshot.py. For the details on what worked with clip and what did not,
refer to that file.
'''

def init():
    global model, device, preprocess
    device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model(modelname, modeltype, weightpath):
    global model, device, preprocess
    print(f"Using {modelname} model {modeltype} version")
# ──  Load Model ─────────────────────────────────────────────
    print(f"Using device: {device}")
    model, preprocess = src.model_utils.load_model(modelname)
    if modeltype == "tuned":
        if weightpath =="":
            raise Exception("Please provide the tuned weights")
        print(f"Loading checkpoint file {weightpath}")
        checkpoint = torch.load(weightpath, map_location=device)
        model.load_state_dict(checkpoint)
    model.eval()


def create_output(filename, similarity, labels):
    filename = str(filename)   # Convert PosixPath to string
    rows = []
    for label, score in zip(labels, similarity[0]):
        label_handler.setScore(label, score)


    dataframe = label_handler.getAsDataframe()
    dataframe["filename"] = filename
    import json
    result = {
        "filename": filename,
        "matches": dataframe.to_dict(orient="records")
    }
    return result

def get_embeddings(image_path, labels):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    texts = clip.tokenize(labels).to(device)
    #print(f"current instrument = {label} index = {idx} available = {available_instruments}")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        # Normalize i.e converting the vector into unit vector
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Similarity
        # Sharpened (divide by temperature before softmax)
        temperature = 0.07
        '''
        We found the image embeddings and text embeddings. Now multiply that
        and ampify the difference.
        '''
        logits = image_features @ text_features.T
        # Softmax different from normalization. converts the probabilities
        # to sum to 1. It is a way to convert the scores into probabilities. The higher the score, the higher the probability.
        '''
        softmax(z_i) = e^z_i / sum(e^z_j)  for j = 1 to n
        '''
        similarity = (logits / temperature).softmax(dim=-1)
    return similarity


def test_images_from_directory(path):
    directory = Path(path)
    '''
    I am not deleting the labels that have been used previously.
    They did not lead to the great results except the latest ones
    in workinglabels. But I am keeping them so that
    we can use them in writing the paper.
    For more details on label also refer bulk_met_zeroshot.py as that contains
    all the functions written and manipulated.
    '''
    #texts = clip.tokenize(available_instruments).to(device)
    # texts = clip.tokenize([
    #         "An image of an Indian miniature painting depicting music making activity. ",
    #         "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
    # ]).to(device)
    # labels_notworking = [
    #     "a person playing a musical instrument",
    #     "a musician playing an instrument",
    #     "a person holding a violin instrument",
    #     "a person playing barrel drum",
    #     "a portrait of a person",
    #     "a person standing without objects",
    #     "a traditional painting of people",
    #     "a drawing of a person without instruments"
    # ]
    # labels_notworking = [
    #     "a person playing violin",           # specific positive
    #     "a person playing a barrel drum",    # specific positive
    #     "a person playing a gourd instrument", # specific positive
    #     "an empty room with no people",      # strong negative
    #     "a person cooking food",             # plausible but wrong
    #     "a landscape photograph",            # strong negative
    # ]
    # These are better than working labels workinglabels
    # another_working_labels = [
    #     'A woman with Indian Lute, Veena',
    #     'Women with musical instrument',
    #     'Woman with a barrel drum',
    #     'a lake',
    #     'a bird',
    #     'bearded man'
    # ]
    workinglabels = [
        "a human playing musical instrument",
        "a person playing a barrel drum",
        "a person playing a gourd instrument",
        "an empty room with no people",
        "an aerial view of a landscape",       # very different visually
        "a close up of food on a plate",       # no person at all
        "inanimate objects with no people present"
    ]
    # Also catch .jpeg extension
    for filepath in [*directory.rglob("*.jpg"), *directory.rglob("*.jpeg")]:
        print(filepath)
        similarity = get_embeddings(filepath, workinglabels)
        create_output(filepath, similarity, workinglabels)

def execute_query(path, labels):
    similarity = get_embeddings(path, labels)
    output = create_output(path, similarity, labels)
    return output

def cleanup():

    global model, preprocess
    del model
    del preprocess
    gc.collect()  # force Python garbage collection

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_help():
    print("Usage:")
    print("  python -m src.clip_inference modelname=")
    print("  python -m src.run training")
    print("  python -m src.run quick-inference")
    print("  python -m src.run quick-training")
    print("  python -m src.run sanity")
    print("  python -m src.run binary-quick-inference")

def run_inference():
    init()
    parser = argparse.ArgumentParser(description="CLIP_inference")
    parser.add_argument("--modelname",
                        type = str,
                        default = "ViT-B/32",
                        help = "Name of the model tobe used. If not present in the local repo, it will be downloaded. Default is `ViT-B/32`")
    parser.add_argument("--modeltype",
                        type = str,
                        default = 'base',
                        help = "Type of the model. [base|tuned]. If `tuned` is used, you must provide weightpath. Default is `base`")

    parser.add_argument("--weightpath",
                        type = str,
                        default = "",
                        help = "Path of the fine-tuned weights.")
    parser.add_argument("--querydatapath",
                        type = str,
                        default = "data/inference/random_images",
                        help ="Path of the folder containing query images. Default is `data/inference/random_images`")

    args = parser.parse_args()
    #params = {k: v for k, v in vars(args).items()}
    load_model(args.modelname, args.modeltype, args.weightpath)
    test_images_from_directory(args.querydatapath)
    cleanup()


if __name__ == "__main__":
    init()
    run_inference()


def run_query(model, flavor, weightpath, labels, query_path):
    global label_handler
    print(f" model = {model} tuned/base = {flavor} labels = {labels} query = {query_path}")
    label_handler = src.web.labels.LabelHandler(labels)
    queriable_labels = label_handler.getQueriableLabels()
    load_model(model, flavor, weightpath)
    output = execute_query(query_path, queriable_labels)
    return output