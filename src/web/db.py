"""
Drishti — MongoDB persistence layer (pymongo + Atlas)
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pandas import col
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure

# ── Connection ────────────────────────────────────────────────────────────────

MONGO_URI = "mongodb+srv://drishti-admin:testpassword123@cluster0.0p2owuz.mongodb.net/?appName=Cluster0"
DB_NAME   = "drishti"
COL_NAME  = "images"
'''
This collection will store documents resulting from 2 step inference.
'''
TWO_STEP_COLLLECTION = "two_step_images"

_client = None
_col    = None

def get_collection():
    global _client, _col
    if _col is None:
        _client = MongoClient(MONGO_URI)
        _col    = _client[DB_NAME][COL_NAME]
    return _col

def get_two_step_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME][TWO_STEP_COLLLECTION]
# ── Checksum ──────────────────────────────────────────────────────────────────

# If 2 files have same content, they will have same SHA-256, and we can use that to avoid duplicate entries in the DB.
def sha256_of_file(path: str | Path) -> str:
    """Return hex SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Write ─────────────────────────────────────────────────────────────────────

def upsert_reading(
    checksum: str,
    filename: str,
    model: str,
    flavor: str,
    labels: list[str],
    matches: list[dict],
    notes: str = "",
) -> str:
    """
    Insert a new inference reading for an image.
    Creates the image document if it doesn't exist yet.
    Returns the checksum (used as _id).

    matches format: [{"label": "...", "score": 0.82}, ...]
    """
    col = get_collection()

    reading = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "model":   model,
        "flavor":  flavor,
        "labels":  labels,
        "matches": matches,
        "notes":   notes,
    }

    col.update_one(
        {"_id": checksum},
        {
            "$setOnInsert": {
                "_id":        checksum,
                "filename":   filename,
                "first_seen": datetime.now(timezone.utc).isoformat(),
            },
            "$push": {"readings": reading},
        },
        upsert=True,
    )
    return checksum

def upsert_reading_to_two_step_collection(
    checksum: str,
    filename: str,
    image_object_path: str,
    model: str,
    flavor: str,
    labels: list[str],
    matches: list[dict],
    notes: str = "",
) -> str:
    col = get_two_step_collection()

    reading = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "model":   model,
        "flavor":  flavor,
        "labels":  labels,
        "matches": matches,
        "notes":   notes,
    }

    col.update_one(
        {"_id": checksum},
        {
            "$setOnInsert": {
                "_id":        checksum,
                "filename":   filename,
                "image_object_path": image_object_path,
                "first_seen": datetime.now(timezone.utc).isoformat(),
            },
            "$push": {"readings": reading},
        },
        upsert=True,
    )
    return checksum


# ── Read ──────────────────────────────────────────────────────────────────────

def get_image(checksum: str) -> dict | None:
    """Return full image document by checksum, or None if not found."""
    col = get_collection()
    return col.find_one({"_id": checksum})

def get_image_from_two_step_collection(checksum: str) -> dict | None:
    """Return full image document by checksum, or None if not found."""
    col = get_two_step_collection()
    return col.find_one({"_id": checksum})


def list_images(limit: int = 50) -> list[dict]:
    """Return most recently seen images (summary only, no readings array)."""
    col = get_collection()
    cursor = col.find(
        {},
        {"filename": 1, "first_seen": 1, "readings": 1}
    ).sort("first_seen", DESCENDING).limit(limit)
    return list(cursor)

def list_images_from_two_step_collection(limit: int = 50) -> list[dict]:
    """Return most recently seen images (summary only, no readings array)."""
    col = get_two_step_collection()
    cursor = col.find(
        {},
        {"filename": 1, "first_seen": 1, "readings": 1}
    ).sort("first_seen", DESCENDING).limit(limit)
    return list(cursor)


# ── Health ────────────────────────────────────────────────────────────────────

def db_connection_check() -> bool:
    try:
        get_collection().database.client.admin.command("ping")
        return True
    except ConnectionFailure:
        return False

def db_connection_close():
    global _client
    if _client is not None:
        _client.close()
        _client = None


def write_to_persistent_storage(collection_name, filename, model, flavor,query_path, response, labels):
    try:
        checksum = sha256_of_file(query_path)
        if collection_name == "two_step_images":
            upsert_reading_to_two_step_collection(
                checksum = checksum,
                filename = filename,
                image_object_path = query_path,
                model    = model,
                flavor   = flavor,
                labels   = labels,
                matches  = response.get("matches", []),
            )
        elif collection_name == "images"    :
            upsert_reading(
                checksum = checksum,
                filename = filename,
                model    = model,
                flavor   = flavor,
                labels   = labels,
                matches  = response.get("matches", []),
            )
            print(f"Saved reading to MongoDB: {checksum[:12]}...")
        return checksum
    except Exception as e:
        print(f"MongoDB write failed: {e}")
        return "Error: MongoWrite failed"
