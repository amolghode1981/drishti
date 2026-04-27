import asyncio
import gc
from json import load
import re
import sys

import clip
import pandas as pd
from sympy import im
import torch
import src.model_utils
from PIL import Image
from pathlib import Path
import argparse
import src.web.labels
from src.gdino import cleanup_output, run_inference_for_file

model = None
device = None
preprocess = None
'''
This is a clean code picked up from bulk_met_zeroshot.py. For the details on what worked with clip and what did not,
refer to that file.
'''

def init():
    global model, device, preprocess
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(modelname, modeltype, weightpath):
    global model, device, preprocess
    print(f"Using {modelname} model {modeltype} version")
# ──  Load Model ─────────────────────────────────────────────
    print(f"Using device: {device}")
    if model is None and preprocess is None:
        model, preprocess = src.model_utils.load_model(modelname)
        if modeltype == "tuned":
            if weightpath =="":
                raise Exception("Please provide the tuned weights")
            print(f"Loading checkpoint file {weightpath}")
            checkpoint = torch.load(weightpath, map_location=device)
            model.load_state_dict(checkpoint)
        model.eval()


def create_output(path, object_filename, similarity, labels):
    object_filename = str(object_filename)   # Convert PosixPath to string
    rows = []
    for label, score in zip(labels, similarity[0]):
        label_handler.setScore(label, score)


    dataframe = label_handler.getAsDataframe()
    dataframe["filename"] = object_filename
    import json
    result = {
        "filename": path,
        "image_object_path": object_filename,
        "matches": dataframe.to_dict(orient="records")
    }
    return result

def get_embeddings(image_path, labels):
    #print(f"Getting embeddings for {image_path}", flush=True)
    image_data = Image.open(image_path)
    if image_data == None:
        print(f"Failed to open image {image_path}", flush=True)
        return None
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


# def test_images_from_directory(path):
#     directory = Path(path)
#     '''
#     I am not deleting the labels that have been used previously.
#     They did not lead to the great results except the latest ones
#     in workinglabels. But I am keeping them so that
#     we can use them in writing the paper.
#     For more details on label also refer bulk_met_zeroshot.py as that contains
#     all the functions written and manipulated.
#     '''
#     #texts = clip.tokenize(available_instruments).to(device)
#     # texts = clip.tokenize([
#     #         "An image of an Indian miniature painting depicting music making activity. ",
#     #         "An image of an Indian miniature painting with no music making activity and it does not have people playing musical instruments "
#     # ]).to(device)
#     # labels_notworking = [
#     #     "a person playing a musical instrument",
#     #     "a musician playing an instrument",
#     #     "a person holding a violin instrument",
#     #     "a person playing barrel drum",
#     #     "a portrait of a person",
#     #     "a person standing without objects",
#     #     "a traditional painting of people",
#     #     "a drawing of a person without instruments"
#     # ]
#     # labels_notworking = [
#     #     "a person playing violin",           # specific positive
#     #     "a person playing a barrel drum",    # specific positive
#     #     "a person playing a gourd instrument", # specific positive
#     #     "an empty room with no people",      # strong negative
#     #     "a person cooking food",             # plausible but wrong
#     #     "a landscape photograph",            # strong negative
#     # ]
#     # These are better than working labels workinglabels
#     # another_working_labels = [
#     #     'A woman with Indian Lute, Veena',
#     #     'Women with musical instrument',
#     #     'Woman with a barrel drum',
#     #     'a lake',
#     #     'a bird',
#     #     'bearded man'
#     # ]
#     workinglabels = [
#         "a human playing musical instrument",
#         "a person playing a barrel drum",
#         "a person playing a gourd instrument",
#         "an empty room with no people",
#         "an aerial view of a landscape",       # very different visually
#         "a close up of food on a plate",       # no person at all
#         "inanimate objects with no people present"
#     ]
#     # Also catch .jpeg extension
#     for filepath in [*directory.rglob("*.jpg"), *directory.rglob("*.jpeg")]:
#         print(filepath)
#         similarity = get_embeddings(filepath, workinglabels)
#         create_output(filepath, similarity, workinglabels)

# def execute_query(path, labels):
#     similarity = get_embeddings(path, labels)
#     output = create_output(path, similarity, labels)
#     return output

# def run_query(model, flavor, weightpath, labels, query_path):
#     global label_handler
#     print(f" model = {model} tuned/base = {flavor} labels = {labels} query = {query_path}")
#     label_handler = src.web.labels.LabelHandler(labels)
#     queriable_labels = label_handler.getQueriableLabels()
#     load_model(model, flavor, weightpath)
#     output = execute_query(query_path, queriable_labels)
#     return output

def cleanup():
    global model, preprocess
    del model
    del preprocess
    gc.collect()  # force Python garbage collection

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def execute_deno_seperator(path):
    workinglabels = [
        "person",
        "human",
        "man",
        "woman"
    ]
    result = asyncio.run(run_inference_for_file(path, workinglabels))
    return result

def execute_CLIP_on_dino_objects(results, labels):
    global label_handler
    label_handler = src.web.labels.LabelHandler(labels)
    queriable_labels = label_handler.getQueriableLabels()
    processed_rows = []
    for result in results:
        path = result["path"]
        object_path = Path(result["output_path"])
        #print(f"Processing object {object_path} from original image {path}")
        similarity = get_embeddings(object_path, queriable_labels)
        result = create_output(path,object_path, similarity, queriable_labels)
        processed_rows.append(result)
    return processed_rows

def run_composite_query(model, flavor, weightpath, labels, query_path):
    init()
    load_model(model, flavor, weightpath)
    '''
    What we are trying to verify here is that if we separate the image objects
    using groundng DINO and then run CLIP on each of those objects, do we see
    a better results in guessing the musical vs non-musical labels.
    '''
    print(f" model = {model} tuned/base = {flavor} labels = {labels} query = {query_path}")
    result = execute_deno_seperator(query_path)
    print(f"Grounding DINO output = {result}")
    clip_result = execute_CLIP_on_dino_objects(result, labels)
    #print(f"CLIP on DINO output = {clip_result}")
    return clip_result

def delete_intermediate_files():
    cleanup_output()
    # global model, preprocess
    # del model
    # del preprocess
    # gc.collect()  # force Python garbage collection

    # if torch.backends.mps.is_available():
    #     torch.mps.empty_cache()
    # elif torch.cuda.is_available():
    #     torch.cuda.empty_cache()


if __name__ == "__main__":
    run_composite_query("ViT-B/32", "base", "", [
        "a human playing musical instrument"], "data/inference/Tests/DP156144.jpg")