import os
from ultralytics import YOLO
import torch


def get_project_root():
    """Get absolute path to project root"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )


def get_data_yaml():
    root = get_project_root()
    yaml_path = os.path.join(root, "data", "processed", "dataset.yaml")

    print(f"📂 Dataset YAML path: {yaml_path}")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ dataset.yaml not found at {yaml_path}")

    return yaml_path


def get_device():
    if torch.cuda.is_available():
        print("🚀 Using GPU")
        return 0
    else:
        print("⚠️ Using CPU")
        return "cpu"


def main():
    print("\n==============================")
    print("🚀 YOLO TRAINING STARTING")
    print("==============================\n")

    # Paths
    data_yaml = get_data_yaml()

    # Device
    device = get_device()

    # Load model
    print("📦 Loading YOLO model...")
    model = YOLO("yolo11n.pt")  # change to yolo11s.pt later

    print("✅ Model loaded\n")

    # Start training
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,

        project=os.path.join(get_project_root(), "outputs"),
        name="yolo11_baseline",
        exist_ok=True,

        # 🔥 solid baseline settings
        optimizer="AdamW",
        lr0=0.01,
        mosaic=1.0,
        mixup=0.1,

        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n==============================")
    print("✅ TRAINING COMPLETE")
    print("==============================\n")

    print("📊 Results:", results)


if __name__ == "__main__":
    main()