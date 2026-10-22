from ast import List
import csv
import asyncio
from email.mime import image
from pathlib import Path

from fsspec import asyn
from httpx import get

image_directory = None
output_directory = None
FFMPEG     = "/Users/amolghode/Documents/work/video_creation/ffmpeg"
counter = 0

processed_output = []
queue = asyncio.Queue()
consumer_task = None

def init(output_dir):
    global queue, consumer_task, image_directory, output_directory, processed_output
    processed_output = []
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    queue = asyncio.Queue()
    consumer_task = asyncio.create_task(command_executor())

def add_to_queue(rows: List):
    for row in rows:
        queue.put_nowait(row)

def stop_queue():
    queue.put_nowait(None)  # Sentinel value to signal completion

async def wait_for_completion():
    await consumer_task
    return processed_output


async def command_executor():
    global counter
    loop = asyncio.get_event_loop()
    tasks = []
    while True:
        row = await queue.get()
        counter += 1
        if row is None:  # Sentinel value to signal completion
            break
        #await run_command(row)
        task = loop.run_in_executor(None, run_command, row, counter)
        tasks.append(task)
        queue.task_done()
    await asyncio.gather(*tasks)


def run_command(row, index):
    import subprocess
    cmd, output_path = get_command_from_row(row, index)
    #print(f"Running command: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        row["image_object_path"] = output_path
        #print(f"✅ Done: {' '.join(cmd)}", flush=True)
    else:
        print(f"❌ Failed: {' '.join(cmd)}", flush=True)
        print(result.stderr.decode(), flush=True)
    processed_output.append(row)

def get_command_from_row(row, index):
    file_path = row['path']
    ffmpeg_string = row['ffmpeg_string']
    output_path   = output_directory / f"{row['filename']}_{index}.jpg"
    #print(f"Generated output path {output_path} index = {index}")
    return [FFMPEG, "-i", str(file_path), "-vf", ffmpeg_string, "-q:v", "2", str(output_path), "-y"], str(output_path)