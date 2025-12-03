# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file to process bird images
"""

import json
import io
import os
import tarfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------
# CONFIG: edit these paths if needed
# -------------------------------------------------------
TRAIN_TAR   = "train_mini.tar.gz"    # path to original mini tar
TRAIN_JSON  = "train_mini.json"   # mini annotations
OUTPUT_DIR  = Path("output_tars")          # folder to save new tar shards
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (96, 96)                     # resize images to 96x96
SHARD_SIZE_BYTES = 2 * (1024**3)           # ~2GB per shard (overkill for mini, but fine)
# -------------------------------------------------------


# -------------------------------------------------------
# STEP 1 — Load annotations and collect bird filenames
# -------------------------------------------------------
print(f"Loading annotations from {TRAIN_JSON}...")
with open(TRAIN_JSON, "r") as f:
    data = json.load(f)

bird_cat_ids = set()
for cat in data["categories"]:
    sc = (cat.get("supercategory") or "").lower()
    cls = (cat.get("class") or "").lower()
    if sc == "aves" or cls == "aves":
        bird_cat_ids.add(cat["id"])

print(f"Found {len(bird_cat_ids)} bird categories.")

# Map image_id -> file_name from JSON
img_id_to_name = {im["id"]: im["file_name"] for im in data["images"]}

# Build set of file_names that are birds
bird_file_names = {
    img_id_to_name[ann["image_id"]]
    for ann in data["annotations"]
    if ann["category_id"] in bird_cat_ids
}

print(f"Total bird images (by annotation): {len(bird_file_names)}")


# -------------------------------------------------------
# Helper: robustly match tar member name to JSON file_name
# -------------------------------------------------------
def match_member_name(member_name, bird_names):
    """
    Try to match a tar member name (which may have extra leading folders)
    to one of the JSON file_name entries in bird_names.

    Example:
      member.name = "train_mini/0001/img.jpg"
      file_name   = "0001/img.jpg"
    We progressively strip leading directories until we find a match.
    """
    # Normalize slashes
    m = member_name.replace("\\", "/")

    # Direct match first
    if m in bird_names:
        return m

    # Progressively strip leading directories:
    # "train_mini/0001/img.jpg" -> "0001/img.jpg" -> "img.jpg"
    parts = m.split("/")
    for i in range(1, len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in bird_names:
            return candidate

    # No match found
    return None


# -------------------------------------------------------
# STEP 2 — Open source tar and write bird images to shards
# -------------------------------------------------------
def open_new_shard(idx):
    """Create a new tar shard inside output folder."""
    shard_path = OUTPUT_DIR / f"only_birds-{idx:04d}.tar"
    tar = tarfile.open(shard_path, "w")
    return tar, shard_path, 0  # current size counter


print(f"Opening source tar: {TRAIN_TAR}")
shard_idx = 0
out_tar, shard_path, curr_size = open_new_shard(shard_idx)

written = 0
skipped = 0
errors = 0

with tarfile.open(TRAIN_TAR, "r:*") as src_tar:
    # List only file entries
    members = [m for m in src_tar if m.isfile()]

    for member in tqdm(members, desc="Streaming from train_mini.tar"):
        # Check if this member is one of the bird images
        match = match_member_name(member.name, bird_file_names)
        if match is None:
            skipped += 1
            continue

        # Read the image bytes from the tar stream
        f = src_tar.extractfile(member)
        if f is None:
            errors += 1
            continue

        try:
            # Load, resize, re-encode as JPEG
            img = Image.open(f).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()

            # Create new tar entry with the matched name
            info = tarfile.TarInfo(name=match)
            info.size = len(img_bytes)

            # Rotate shard if it would exceed the shard size
            if curr_size + info.size + 512 > SHARD_SIZE_BYTES:
                out_tar.close()
                shard_idx += 1
                out_tar, shard_path, curr_size = open_new_shard(shard_idx)

            out_tar.addfile(info, io.BytesIO(img_bytes))
            curr_size += info.size + 512
            written += 1

        except Exception as e:
            errors += 1
            # Uncomment for debugging:
            # print(f"Error processing {member.name}: {e}")

out_tar.close()

print("\n✅ DONE.")
print(f"Wrote {written} bird images into {shard_idx+1} shard(s) inside {OUTPUT_DIR}/")
print(f"Skipped non-bird files: {skipped}")
print(f"Errors on read/resize:   {errors}")
