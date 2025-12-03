from pathlib import Path
from PIL import Image

SRC_ROOT = Path.home() / "data" / "coco" / "images"
DST_ROOT = Path.home() / "data" / "coco96"
SPLITS = ["train2017"]  # only train

size = (96, 96)

for split in SPLITS:
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(src_dir.glob("*.jpg"))
    print(f"{split}: found {len(img_paths)} images")

    for i, img_path in enumerate(img_paths, 1):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize(size, Image.BILINEAR)
                out_path = dst_dir / img_path.name
                im.save(out_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"Failed on {img_path}: {e}")

        if i % 1000 == 0:
            print(f"{split}: processed {i}/{len(img_paths)}")

print("Done.")
