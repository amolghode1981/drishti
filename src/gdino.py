from cProfile import label
from importlib.resources import path
from pathlib import Path
import token


from huggingface_hub import try_to_load_from_cache
import asyncio
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import torch
from PIL import Image
import pandas

model = None
device = None
preprocess = None
from src.web.object_separation_ffmpeg import init as ffmpeg_init, add_to_queue, stop_queue, wait_for_completion
_MODELS = {
    "swint": "IDEA-Research/grounding-dino-tiny",
    "swinb": "IDEA-Research/grounding-dino-base",
}
def init():
    global model, device, preprocess
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    ffmpeg_init("data/cropped")


def load_model(repo_id):
    global model, preprocess
    if model is None and preprocess is None:
        preprocess = AutoProcessor.from_pretrained(_MODELS[repo_id])
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODELS[repo_id]).to(device)
        model.eval()

def create_output(path: Path, results, labels):
    print(f"create_output Creating output for {path.name} with labels {labels}", flush=True)
    result = results[0]
    filename = str(path.name)   # Convert PosixPath to string
    if len(result["boxes"]) == 0:
        row = {
            'path': str(path),
            'filename': filename,
            'type': 'miniature',
            'labels':'None',
            'confidence': '-1',
            'ffmpeg_string': ""
        }
    else:
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            row = {
                'path': str(path),
                'filename': filename,
                'type': 'miniature',
            }
            box = [round(x, 2) for x in box.tolist()]
            padding = 50
            x1 = max(0, int(box[0]) - padding)
            y1 = max(0, int(box[1]) - padding)
            w  = int(box[2] - box[0]) + padding * 2
            h  = int(box[3] - box[1]) + padding * 2
            ffmpeg_string = f"crop={w}:{h}:{x1}:{y1}"
            #ffmpeg_string = f"crop={int(box[2] - box [0])}:{int(box[3] - box[1])}:{int(box[0])}:{int(box[1])}"
            #dims = [int(box[0]), int(box[1]), int(box[2] - box [0]), int(box[3] - box[1])]
            row['labels'] = labels
            row['confidence'] = round(score.item(), 3)
            row['ffmpeg_string'] = ffmpeg_string
            # print(f"Adding row for {filename} with labels {labels} and confidence {score.item()} and ffmpeg string {ffmpeg_string}", flush=True)
            add_to_queue([row])

def get_embeddings(image_path, labels, box_threshold=0.3, text_threshold=0.25):
    # print("get_embeddings", flush=True)
    image = Image.open(image_path).convert("RGB")
    inputs = preprocess(text=labels, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = preprocess.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )
    return results

def detect_objects_in_image(image_path, labels: list):
    text_prompt = " . ".join(labels) + " ."
    print(text_prompt)
    results = get_embeddings(image_path, text_prompt)
    result = results[0]
    # for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
    #     print(f"[{score:.2f}] {label} box={[round(x,2) for x in box.tolist()]}")
    create_output(image_path, results, text_prompt)
'''
manage data clearly.
1. Associate each cropped image with the label. : Done
2. Use that label with clip inference to see if there is music. : Done
3. Run ffmpeg from python to crop the original image and save the cropped image with the label as the name. : Done
4. create clip_grounding_dino_inference.py that would first separate objects using grounding dino and then run clip inference on the cropped images to classify them as music or non-musiic
'''
async def run_inference_for_file(filepath: str, labels: list):
    init()
    load_model("swinb")

    detect_objects_in_image(Path(filepath), labels)
    stop_queue()
    rows = await wait_for_completion()
    return rows

def cleanup_output():
    output_directory = Path("data/cropped")
    for file in output_directory.glob("*.jpg"):
        file.unlink()
    for file in output_directory.glob("*.jpeg"):
        file.unlink()