import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
import cv2

# ===== PATHS =====
BASE_PATH = "yolo_pipeline/data/raw/occlusion-dataset"
PROCESSED_PATH = "yolo_pipeline/data/processed"

SAMPLES_PATH = os.path.join(BASE_PATH, "samples")
LABELS_PATH = os.path.join(BASE_PATH, "labels")
SPLITS_PATH = os.path.join(BASE_PATH, "splits")

# ===== SETTINGS =====
CAMERA_KEYWORD = "front"   # change later for multi-view
CLASS_MAP = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2
}


def get_image_from_frame(frame_path):
    """
    Pick one image from frame folder.
    Prioritize 'front' camera if available.
    """
    files = os.listdir(frame_path)

    images = [f for f in files if f.endswith(".jpg")]

    if len(images) == 0:
        return None

    # prioritize front camera
    for img in images:
        if CAMERA_KEYWORD in img:
            return os.path.join(frame_path, img)

    # fallback: first image
    return os.path.join(frame_path, images[0])


def convert_bbox(size, box):
    """
    Convert (x1,y1,x2,y2) → YOLO (x_center, y_center, w, h)
    """
    w, h = size

    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h

    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return x_center, y_center, bw, bh


def process_split(split):
    print(f"\nProcessing {split}...")

    labels_file = os.path.join(LABELS_PATH, f"{split}_labels.csv")
    samples_file = os.path.join(SPLITS_PATH, f"{split}_samples.json")

    df = pd.read_csv(labels_file)

    with open(samples_file, "r") as f:
        frame_ids = set(json.load(f))

    output_img_dir = os.path.join(PROCESSED_PATH, "images", split)
    output_lbl_dir = os.path.join(PROCESSED_PATH, "labels", split)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    grouped = df.groupby("frame_id")

    for frame_id, group in tqdm(grouped):
        if frame_id not in frame_ids:
            continue

        frame_path = os.path.join(SAMPLES_PATH, split, frame_id)

        if not os.path.exists(frame_path):
            continue

        img_path = get_image_from_frame(frame_path)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # copy image
        out_img_path = os.path.join(output_img_dir, f"{frame_id}.jpg")
        shutil.copy(img_path, out_img_path)

        # create label file
        label_lines = []

        for _, row in group.iterrows():
            cls = row["object_class"]

            if cls not in CLASS_MAP:
                continue

            bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
            x, y, bw, bh = convert_bbox((w, h), bbox)

            class_id = CLASS_MAP[cls]

            label_lines.append(f"{class_id} {x} {y} {bw} {bh}")

        out_lbl_path = os.path.join(output_lbl_dir, f"{frame_id}.txt")

        with open(out_lbl_path, "w") as f:
            f.write("\n".join(label_lines))


def create_yaml():
    yaml_path = os.path.join(PROCESSED_PATH, "dataset.yaml")

    content = f"""
path: {os.path.abspath(PROCESSED_PATH)}
train: images/train
val: images/val

names:
  0: car
  1: pedestrian
  2: cyclist
"""

    with open(yaml_path, "w") as f:
        f.write(content.strip())


def main():
    process_split("train")
    process_split("val")
    create_yaml()

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()