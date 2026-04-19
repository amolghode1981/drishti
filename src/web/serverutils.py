import os
from pathlib import Path
import base64, mimetypes

STORAGE_DIR = "data/inference/query_images"
def read_image_from_storage(filename: str) -> bytes:
    """Read image bytes from storage by filename."""
    bytes = None
    path = os.path.join(STORAGE_DIR, filename)
    with open(path, "rb") as f:
        bytes = f.read()
    if bytes is not None:
        mime, _ = mimetypes.guess_type(path)
        return f"data:{mime};base64,{base64.b64encode(bytes).decode()}"
    else:
        raise FileNotFoundError(f"Image {filename} not found in storage.")

def write_image_to_storage(filename: str, data: bytes) -> str:
    """Write image bytes to storage and return the path."""
    save_dir = Path(STORAGE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    query_path = save_dir / filename
    with open(query_path, "wb") as f:
        f.write(data)
    return str(query_path)
