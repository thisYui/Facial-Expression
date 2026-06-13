import argparse
import csv
import json
import sys
from pathlib import Path


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
    "suprise": 0,
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

RAF_INT_TO_CAFE = {
    1: 0,  # surprise
    2: 1,  # fear
    3: 2,  # disgust
    4: 3,  # happy
    5: 4,  # sadness
    6: 5,  # anger
    7: 6,  # neutral
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DATASET_DEFAULTS = {
    "rafdb": {
        "base_subdirs": ["DATASET", "."],
        "split": "test",
        "label_scheme": "raf_int",
        "note": "RAF-DB folder layout uses numeric class folders 1..7.",
    },
    "ferplus": {
        "base_subdirs": ["."],
        "split": "test",
        "label_scheme": "names",
        "note": "FERPlus has train/validation/test class folders; contempt is skipped.",
    },
    "affectnet": {
        "base_subdirs": ["archive (3)", "."],
        "split": "Test",
        "label_scheme": "names",
        "note": "AffectNet Kaggle input nests data under archive (3); contempt is skipped.",
    },
    "sfew": {
        "base_subdirs": ["."],
        "split": "Val",
        "label_scheme": "names",
        "note": "SFEW Test/Test_Aligned_Faces is unlabeled; Val is the labeled evaluation split.",
    },
    "mma": {
        "base_subdirs": ["MMAFEDB", "."],
        "split": "test",
        "label_scheme": "names",
        "note": "MMAFEDB has train/valid/test class folders.",
    },
}


def import_cafe_model():
    """Import official CAFE code without letting its argparse consume our flags."""
    original_argv = sys.argv[:]
    sys.argv = ["ours_CAFE.py"]
    try:
        import ours_CAFE
    finally:
        sys.argv = original_argv
    return ours_CAFE.Model, ours_CAFE.clip_model


def normalize_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def label_to_cafe(label, label_scheme):
    normalized = normalize_name(label)
    if normalized == "contempt":
        return None, "contempt"

    if label_scheme == "raf_int":
        if not normalized.isdigit():
            raise ValueError(f"Expected RAF numeric label folder, got {label!r}")
        label_id = int(normalized)
        if label_id not in RAF_INT_TO_CAFE:
            raise ValueError(f"Unsupported RAF label folder {label!r}; expected 1..7")
        cafe_label = RAF_INT_TO_CAFE[label_id]
        return cafe_label, CAFE_CLASS_NAMES[cafe_label]

    if normalized not in NAME_TO_CAFE:
        raise ValueError(f"Unsupported class folder {label!r}")
    cafe_label = NAME_TO_CAFE[normalized]
    return cafe_label, CAFE_CLASS_NAMES[cafe_label]


def resolve_child_case_insensitive(parent, child_name):
    direct = parent / child_name
    if direct.exists():
        return direct
    target = child_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return direct


def resolve_base_dir(root, dataset_name, base_subdir=None):
    root = Path(root)
    if base_subdir:
        candidate = root / base_subdir
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Cannot find base_subdir {base_subdir!r} under {root}")

    for candidate_name in DATASET_DEFAULTS[dataset_name]["base_subdirs"]:
        candidate = root if candidate_name == "." else root / candidate_name
        if candidate.exists():
            return candidate
    return root


class FolderSplitDataset:
    def __init__(self, root, dataset_name, split, transform, base_subdir=None):
        self.dataset_name = dataset_name
        self.root = Path(root)
        self.base_dir = resolve_base_dir(self.root, dataset_name, base_subdir)
        self.split = split
        self.transform = transform
        self.label_scheme = DATASET_DEFAULTS[dataset_name]["label_scheme"]
        self.eval_dir = resolve_child_case_insensitive(self.base_dir, split)
        self.samples = []
        self.skipped_classes = {}
        self.class_counts = {}
        self._discover()

        if not self.samples:
            raise RuntimeError(
                f"No labeled samples found in {self.eval_dir}. "
                "For SFEW, use --split Val because Test/Test_Aligned_Faces is unlabeled."
            )

    def _discover(self):
        if not self.eval_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.eval_dir}")

        class_dirs = sorted(path for path in self.eval_dir.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            cafe_label, label_name = label_to_cafe(class_dir.name, self.label_scheme)
            image_paths = [
                path for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]

            if cafe_label is None:
                self.skipped_classes[class_dir.name] = len(image_paths)
                continue

            self.class_counts[label_name] = self.class_counts.get(label_name, 0) + len(image_paths)
            for image_path in image_paths:
                self.samples.append({
                    "path": image_path,
                    "label": cafe_label,
                    "label_name": label_name,
                })

    def describe(self):
        return {
            "dataset": self.dataset_name,
            "root": str(self.root),
            "base_dir": str(self.base_dir),
            "split": self.split,
            "eval_dir": str(self.eval_dir),
            "class_counts": self.class_counts,
            "skipped_classes": self.skipped_classes,
            "num_samples": len(self.samples),
            "note": DATASET_DEFAULTS[self.dataset_name]["note"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        return self.transform(image), sample["label"], str(sample["path"])


def per_class_accuracy(y_true, y_pred, labels):
    result = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            continue
        result[CAFE_CLASS_NAMES[label]] = float((y_pred[mask] == label).mean())
    return result


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [CAFE_CLASS_NAMES[label] for label in labels]
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_metrics(dataset, y_true, y_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    true_labels = sorted(set(y_true.tolist()))
    matrix_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_acc = per_class_accuracy(y_true, y_pred, true_labels)
    return {
        "dataset": dataset.dataset_name,
        "dataset_mode": f"{dataset.dataset_name}-{dataset.split}-7cls-no-contempt",
        "num_samples": int(len(dataset)),
        "labels": [CAFE_CLASS_NAMES[label] for label in true_labels],
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_accuracy": float(np.mean(list(class_acc.values()))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=true_labels, average="macro", zero_division=0)),
        "per_class_accuracy": class_acc,
        "class_counts": dataset.class_counts,
        "skipped_classes": dataset.skipped_classes,
        "confusion_matrix_labels": [CAFE_CLASS_NAMES[label] for label in matrix_labels],
        "layout": dataset.describe(),
    }, true_labels, matrix_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAFE on paper cross-dataset FER folder layouts.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--data_path", required=True, help="Dataset root, e.g. /kaggle/working/data/ferplus")
    parser.add_argument("--split", default=None, help="Override evaluation split. Defaults depend on --dataset.")
    parser.add_argument("--base_subdir", default=None, help="Optional subdirectory under data_path to treat as dataset root.")
    parser.add_argument("--checkpoint", default="../../outputs/rafdb_cafe_seed3407/ours_best.pth")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Only print discovered layout and sample counts.")
    args = parser.parse_args()

    split = args.split or DATASET_DEFAULTS[args.dataset]["split"]
    output_dir = Path(args.output_dir or f"../../outputs/{args.dataset}_{split.lower()}_eval")

    if args.dry_run:
        transform = lambda image: image
    else:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    dataset = FolderSplitDataset(args.data_path, args.dataset, split, transform, args.base_subdir)

    if args.dry_run:
        print(json.dumps(dataset.describe(), indent=2))
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

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
    metrics, true_labels, matrix_labels = build_metrics(dataset, y_true, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_true, y_pred, matrix_labels, output_dir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
