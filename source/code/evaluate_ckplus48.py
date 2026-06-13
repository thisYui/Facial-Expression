import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CAFE_CLASS_NAMES = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sadness",
    5: "anger",
    6: "neutral",
}

NAME_TO_CAFE = {
    "surprise": 0,
    "surprised": 0,
    "fear": 1,
    "fearful": 1,
    "disgust": 2,
    "disgusted": 2,
    "happy": 3,
    "happiness": 3,
    "sad": 4,
    "sadness": 4,
    "anger": 5,
    "angry": 5,
    "neutral": 6,
}

# CK+ extended CSV label order used by the Kaggle ckdataset:
# 0 Anger, 1 Disgust, 2 Fear, 3 Happiness, 4 Sadness,
# 5 Surprise, 6 Neutral, 7 Contempt.
CKPLUS_INT_TO_NAME = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sadness",
    5: "surprise",
    6: "neutral",
    7: "contempt",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def import_cafe_model():
    """Import official CAFE code without letting its argparse consume our flags."""
    original_argv = sys.argv[:]
    sys.argv = ["ours_CAFE.py"]
    try:
        import ours_CAFE
    finally:
        sys.argv = original_argv
    return ours_CAFE.Model, ours_CAFE.clip_model


def normalize_label(raw_label):
    if isinstance(raw_label, str):
        value = raw_label.strip().lower()
        if value == "":
            raise ValueError("Empty label")
        if value.isdigit():
            value = int(value)
        else:
            return value
    else:
        value = int(raw_label)

    return CKPLUS_INT_TO_NAME.get(value, str(value))


def label_name_to_cafe(label_name, exclude_neutral=False):
    label_name = label_name.strip().lower()
    if label_name == "contempt":
        return None
    if exclude_neutral and label_name == "neutral":
        return None
    if label_name not in NAME_TO_CAFE:
        raise ValueError(f"Unsupported CK+48 label: {label_name}")
    return NAME_TO_CAFE[label_name]


def pil_from_pixels(pixel_text):
    values = np.fromstring(str(pixel_text), sep=" ", dtype=np.uint8)
    side = int(np.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"Pixel vector length is not square: {values.size}")
    image = values.reshape(side, side)
    return Image.fromarray(image, mode="L").convert("RGB")


def pil_from_path(path):
    return Image.open(path).convert("RGB")


class CKPlus48Dataset(Dataset):
    def __init__(self, root, transform, exclude_neutral=False):
        self.root = Path(root)
        self.transform = transform
        self.exclude_neutral = exclude_neutral
        self.samples = self._discover_samples()
        if not self.samples:
            raise RuntimeError(f"No usable CK+48 samples found under {self.root}")

        labels = sorted({sample["label"] for sample in self.samples})
        self.mode = "CK+48-6cls-peak-only" if exclude_neutral else (
            "CK+48-7cls-with-neutral" if 6 in labels else "CK+48-6cls"
        )
        self.labels = labels

    def _discover_samples(self):
        csv_path = self.root / "ckextended.csv"
        if csv_path.exists():
            return self._samples_from_csv(csv_path)
        return self._samples_from_folders()

    def _samples_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        columns = {str(col).lower(): col for col in df.columns}

        label_col = (
            columns.get("emotion")
            or columns.get("label")
            or columns.get("class")
            or columns.get("target")
        )
        pixel_col = columns.get("pixels") or columns.get("pixel")
        path_col = (
            columns.get("path")
            or columns.get("image")
            or columns.get("filename")
            or columns.get("file")
        )

        if label_col is None:
            raise ValueError(f"Cannot find label column in {csv_path}: {list(df.columns)}")
        if pixel_col is None and path_col is None:
            raise ValueError(f"Cannot find pixels or image path column in {csv_path}: {list(df.columns)}")

        samples = []
        for row_idx, row in df.iterrows():
            label_name = normalize_label(row[label_col])
            cafe_label = label_name_to_cafe(label_name, self.exclude_neutral)
            if cafe_label is None:
                continue

            if pixel_col is not None and not pd.isna(row[pixel_col]):
                samples.append({
                    "id": f"row_{row_idx}",
                    "label": cafe_label,
                    "label_name": CAFE_CLASS_NAMES[cafe_label],
                    "source": "pixels",
                    "pixels": row[pixel_col],
                })
            else:
                image_path = self.root / str(row[path_col])
                samples.append({
                    "id": str(image_path),
                    "label": cafe_label,
                    "label_name": CAFE_CLASS_NAMES[cafe_label],
                    "source": "path",
                    "path": image_path,
                })
        return samples

    def _samples_from_folders(self):
        samples = []
        for class_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            cafe_label = label_name_to_cafe(normalize_label(class_dir.name), self.exclude_neutral)
            if cafe_label is None:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                samples.append({
                    "id": str(image_path),
                    "label": cafe_label,
                    "label_name": CAFE_CLASS_NAMES[cafe_label],
                    "source": "path",
                    "path": image_path,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample["source"] == "pixels":
            image = pil_from_pixels(sample["pixels"])
        else:
            image = pil_from_path(sample["path"])

        return self.transform(image), sample["label"], sample["id"]


def per_class_accuracy(y_true, y_pred, labels):
    result = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            continue
        result[CAFE_CLASS_NAMES[label]] = float((y_pred[mask] == label).mean())
    return result


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [CAFE_CLASS_NAMES[label] for label in labels]
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAFE on CK+48.")
    parser.add_argument("--ck_path", default="../../data/ckplus48", help="Path to CK+48 root")
    parser.add_argument("--checkpoint", default="../../outputs/rafdb_cafe_seed3407/ours_best.pth")
    parser.add_argument("--output_dir", default="../../outputs/ckplus48_eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--exclude_neutral",
        action="store_true",
        help="Evaluate CK+48 as peak-only by dropping neutral and contempt samples.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = CKPlus48Dataset(args.ck_path, transform, exclude_neutral=args.exclude_neutral)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Calculate and display class distribution
    class_counts = {}
    for sample in dataset.samples:
        lbl_name = sample["label_name"]
        class_counts[lbl_name] = class_counts.get(lbl_name, 0) + 1

    print("\n--- Dataset Class Distribution ---")
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count}")
    print(f"Total samples: {len(dataset)}\n")

    # Save class distribution plot
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    names = sorted(class_counts.keys())
    counts = [class_counts[n] for n in names]
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Class Distribution - {dataset.mode}')
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "class_distribution.png", dpi=200)
    plt.close()

    Model, clip_model = import_cafe_model()
    model = Model()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    clip_model.to(device)
    clip_model.eval()

    y_true = []
    y_pred = []
    rows = []

    with torch.no_grad():
        for images, labels, sample_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images, clip_model, labels, phase="test")
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for sample_id, target, pred, prob in zip(sample_ids, labels.cpu().tolist(), preds.cpu().tolist(), probs):
                rows.append({
                    "sample_id": sample_id,
                    "target": target,
                    "target_name": CAFE_CLASS_NAMES[target],
                    "prediction": pred,
                    "prediction_name": CAFE_CLASS_NAMES[pred],
                    "confidence": float(prob[pred]),
                    "correct": bool(target == pred),
                })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(set(y_true.tolist()))
    class_acc = per_class_accuracy(y_true, y_pred, labels)
    mean_accuracy = float(np.mean(list(class_acc.values())))

    metrics = {
        "dataset_mode": dataset.mode,
        "num_samples": int(len(dataset)),
        "labels": [CAFE_CLASS_NAMES[label] for label in labels],
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_accuracy": mean_accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_accuracy": class_acc,
        "class_counts": class_counts,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_true, y_pred, labels, output_dir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
