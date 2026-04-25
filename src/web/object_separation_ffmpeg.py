from ast import List
import csv
import asyncio
from pathlib import Path

from fsspec import asyn
from httpx import get

image_directory = None
output_directory = Path("cropped")
FFMPEG     = "/Users/amolghode/Documents/work/video_creation/ffmpeg"
counter = 0

processed_output = []
queue = asyncio.Queue()
consumer_task = None

def init(images_dir, output_dir):
    global queue, consumer_task, image_directory, output_directory
    image_directory = Path(images_dir)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    queue = asyncio.Queue()
    consumer_task = asyncio.create_task(command_executor())

def add_to_queue(rows: List):
    print("AMOL_DEBUG_add_to_queue", flush=True)
    for row in rows:
        queue.put_nowait(row)
    print("AMOL_DEBUG_add_to_queue_done", flush=True)

def stop_queue():
    queue.put_nowait(None)  # Sentinel value to signal completion

async def wait_for_completion():
    await consumer_task
    return processed_output


async def command_executor():
    loop = asyncio.get_event_loop()
    tasks = []
    while True:
        row = await queue.get()
        if row is None:  # Sentinel value to signal completion
            break
        #await run_command(row)
        task = loop.run_in_executor(None, run_command, row)
        tasks.append(task)
        queue.task_done()
    await asyncio.gather(*tasks)


def run_command(row):
    import subprocess
    global counter
    cmd, output_path = get_command_from_row(row)
    counter += 1
    print(f"Running command: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        row["output_path"] = output_path
        print(f"✅ Done: {' '.join(cmd)}", flush=True)
    else:
        print(f"❌ Failed: {' '.join(cmd)}", flush=True)
        print(result.stderr.decode(), flush=True)
    processed_output.append(row)

def get_command_from_row(row):
    global counter
    path = row['path']
    ffmpeg_string = row['ffmpeg_string']
    image_path    = image_directory / path
    output_path   = output_directory / f"{row['path']}_{counter}.jpg"
    return [FFMPEG, "-i", str(image_path), "-vf", ffmpeg_string, "-q:v", "2", str(output_path), "-y"], str(output_path)