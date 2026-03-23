import os
import shutil
import json
import pandas as pd
from tqdm import tqdm
import cv2

# ===== PATHS =====
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/raw/occlusion-dataset")
)
PROCESSED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed")
)

SAMPLES_PATH = os.path.join(BASE_PATH, "samples")
LABELS_PATH = os.path.join(BASE_PATH, "labels")

# ===== CLASS MAP =====
CLASS_MAP = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2
}

def load_manifest(split):
    manifest_path = os.path.join(SAMPLES_PATH, f"{split}_manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    frame_map = {}

    # 🔥 handle dict-style manifest
    for frame_id, info in manifest.items():
        folder = info["folder"]
        frame_map[frame_id] = os.path.join(SAMPLES_PATH, split, folder)
    print(type(manifest))
    print(list(manifest.keys())[:3])
    return frame_map

def get_image_from_frame(frame_path):
    """
    Get ANY image from frame (no assumptions about naming)
    """
    files = os.listdir(frame_path)
    images = [f for f in files if f.lower().endswith(".jpg")]

    if len(images) == 0:
        return None

    return os.path.join(frame_path, images[0])


def convert_bbox(size, box):
    """
    Convert (x1,y1,x2,y2) → YOLO format
    """
    w, h = size
    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return x_center, y_center, bw, bh


def build_frame_index(split):
    """
    Create mapping: frame_id -> actual folder path
    """
    split_path = os.path.join(SAMPLES_PATH, split)

    frame_map = {}

    for root, dirs, files in os.walk(split_path):
        for d in dirs:
            frame_map[d] = os.path.join(root, d)

    return frame_map


def process_split(split):
    print(f"\nProcessing {split}...")

    labels_file = os.path.join(LABELS_PATH, f"{split}_labels.csv")
    df = pd.read_csv(labels_file)

    output_img_dir = os.path.join(PROCESSED_PATH, "images", split)
    output_lbl_dir = os.path.join(PROCESSED_PATH, "labels", split)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    grouped = df.groupby("frame_id")

    print("🔍 Loading manifest...")
    frame_map = load_manifest(split)
    print(f"✅ Found {len(frame_map)} mapped frames")

    saved_count = 0

    for frame_id, group in tqdm(grouped):
        if frame_id not in frame_map:
            continue

        frame_path = frame_map[frame_id]

        if not os.path.exists(frame_path):
            continue

        img_path = get_image_from_frame(frame_path)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        out_img_path = os.path.join(output_img_dir, f"{frame_id}.jpg")
        shutil.copy(img_path, out_img_path)

        label_lines = []

        for _, row in group.iterrows():
            cls = row["object_class"]

            if cls not in CLASS_MAP:
                continue

            bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
            x, y, bw, bh = convert_bbox((w, h), bbox)

            class_id = CLASS_MAP[cls]

            # 🔥 skip invalid boxes
            if bw <= 0 or bh <= 0:
                continue

            if len(label_lines) == 0:
                continue

            label_lines.append(f"{class_id} {x} {y} {bw} {bh}")


        out_lbl_path = os.path.join(output_lbl_dir, f"{frame_id}.txt")

        with open(out_lbl_path, "w") as f:
            f.write("\n".join(label_lines))

        saved_count += 1

    print(f"✅ Saved {saved_count} images for {split}")


def create_yaml():
    yaml_path = os.path.join(PROCESSED_PATH, "dataset.yaml")

    content = f"""
path: {PROCESSED_PATH}
train: images/train
val: images/val

names:
  0: car
  1: pedestrian
  2: cyclist
"""

    with open(yaml_path, "w") as f:
        f.write(content.strip())

    print("✅ dataset.yaml created")


def main():
    print("🚀 Starting conversion...\n")

    process_split("train")
    process_split("val")
    create_yaml()

    print("\n🎉 Conversion COMPLETE!")


if __name__ == "__main__":
    main()