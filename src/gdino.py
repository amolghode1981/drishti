from cProfile import label
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ffmpeg_init("data/inference/Tests/", "data/inference/Tests/cropped")


def load_model(repo_id):
    global model, preprocess
    preprocess = AutoProcessor.from_pretrained(_MODELS[repo_id], local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODELS[repo_id]).to(device)
    model.eval()

def create_output(filename, results, labels):
    print(f"create_output Creating output for {filename} with labels {labels}", flush=True)
    result = results[0]
    filename = str(filename)   # Convert PosixPath to string
    if len(result["boxes"]) == 0:
        row = {
            'path': filename,
            'type': 'miniature',
            'labels':'None',
            'confidence': '-1',
            'ffmpeg_string': ""
        }
    else:
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            row = {
                'path': filename,
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
            print(f"Adding row for {filename} with labels {labels} and confidence {score.item()} and ffmpeg string {ffmpeg_string}", flush=True)
            add_to_queue([row])

def get_embeddings(image_path, labels, token_spans, box_threshold=0.3, text_threshold=0.25):
    print("get_embeddings", flush=True)
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

def get_token_spans(text_prompt, labels):
    spans = []
    for label in labels:
        start = text_prompt.index(label)
        end = start + len(label)
        spans.append((start, end))
    return spans

async def test_images_from_directory(path):
    directory = Path(path)
    # workinglabels = [
    #     "Veena without any shields or swords nearby and performing music",
    #     "A woman or a man with Indian Lute, Veena",
    #     "People playing music in a courtyard",
    #     "A person playing musical drums",
    #     "A person playing an Indian stringed instrument like Veena or Sitar",
    #     "A person holding musical instruments while performing music",
    #     "A court scene with one or many musical instruments",
    #     "A court scene with musicians and dancers entertaining people",
    #     "A man practicing music alone holding a long necked gaurd lute"
    # ]
    # notworkinglabels = [
    #     "veena",
    #     "sitar",
    #     "lute",
    #     "tabla",
    #     "drum",
    #     "musician",
    #     "a person holding musical instrument",
    #     "a person holding objects",
    #     "dancer",
    #     "stringed instrument",
    #     "musical instrument",
    # ]
    # workinglabels_partial_1 = [
    #     "drum",
    #     "musician",
    #     "person",
    #     "dancer",
    #     "musical instrument",
    # ]
    # workinglabels = [
    #     "drum",
    #     "person",
    #     "musical instrument",
    # ]
    workinglabels = [
        "person",
        "human",
        "man",
        "woman"
    ]
    text_prompt = " . ".join(workinglabels) + " ."
    token_spans = get_token_spans(text_prompt, workinglabels)
    print(text_prompt)
    # Also catch .jpeg extension
    count = 0
    for filepath in [*directory.glob("*.jpg"), *directory.glob("*.jpeg")]:
        if count >= 6:
            break
        count += 1
        print(filepath)
        results = get_embeddings(filepath, text_prompt, token_spans)
        # result = results[0]
        # for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        #     print(f"[{score:.2f}] {label} box={[round(x,2) for x in box.tolist()]}")
        create_output(filepath.name, results, text_prompt)

    stop_queue()
    rows = await wait_for_completion()
    dataframe = pandas.DataFrame(rows)
    dataframe.to_csv("data/inference/Tests/gdino_output.csv", index=False)
'''
manage data clearly.
1. Associate each cropped image with the label. : Done
2. Use that label with clip inference to see if there is music. : Done
3. Run ffmpeg from python to crop the original image and save the cropped image with the label as the name. : Done
4. create clip_grounding_dino_inference.py that would first separate objects using grounding dino and then run clip inference on the cropped images to classify them as music or non-musiic

'''

async def run_inference():
    init()
    load_model("swint")
    await test_images_from_directory("data/inference/Tests/")

if __name__ == "__main__":
    asyncio.run(run_inference())
