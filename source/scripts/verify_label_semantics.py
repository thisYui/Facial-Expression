import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RAF_LABELS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

CKPLUS_LABELS = {
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


def read_image(path, size):
    image = Image.open(path).convert("RGB")
    return image.resize((size, size))


def read_pixels(pixel_text, size):
    values = np.fromstring(str(pixel_text), sep=" ", dtype=np.uint8)
    side = int(math.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"Pixel vector length is not square: {values.size}")
    image = Image.fromarray(values.reshape(side, side), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def make_contact_sheet(samples, title, output_path, image_size=112, cols=5):
    if not samples:
        return

    rows = math.ceil(len(samples) / cols)
    label_h = 32
    title_h = 32
    width = cols * image_size
    height = title_h + rows * (image_size + label_h)

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")

    for idx, (image, caption) in enumerate(samples):
        x = (idx % cols) * image_size
        y = title_h + (idx // cols) * (image_size + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + image_size + 3), caption[:22], fill="black")

    sheet.save(output_path)


def verify_raf(raf_path, output_dir, num_per_class, image_size):
    raf_path = Path(raf_path)
    label_file = raf_path / "EmoLabel" / "list_patition_label.txt"
    image_dir = raf_path / "Image" / "aligned"
    if not label_file.exists():
        raise FileNotFoundError(label_file)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    df = pd.read_csv(label_file, sep=" ", header=None, names=["image", "label"])
    counts = df["label"].value_counts().sort_index().to_dict()
    report = {
        "label_file": str(label_file),
        "image_dir": str(image_dir),
        "num_rows": int(len(df)),
        "counts": {RAF_LABELS.get(int(k), str(k)): int(v) for k, v in counts.items()},
        "missing_images": [],
    }

    for label_id, label_name in RAF_LABELS.items():
        subset = df[df["label"] == label_id].head(num_per_class)
        samples = []
        for _, row in subset.iterrows():
            stem = Path(str(row["image"])).stem
            image_path = image_dir / f"{stem}_aligned.jpg"
            if not image_path.exists():
                report["missing_images"].append(str(image_path))
                continue
            image = read_image(image_path, image_size)
            caption = Path(row["image"]).name
            samples.append((image, caption))
        make_contact_sheet(samples, f"RAF label {label_id}: {label_name}", output_dir / f"raf_{label_id}_{label_name}.png")

    return report


def verify_ckplus(ck_path, output_dir, num_per_class, image_size):
    ck_path = Path(ck_path)
    csv_path = ck_path / "ckextended.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    columns = {str(col).lower(): col for col in df.columns}
    label_col = columns.get("emotion") or columns.get("label") or columns.get("class") or columns.get("target")
    pixel_col = columns.get("pixels") or columns.get("pixel")
    if label_col is None or pixel_col is None:
        raise ValueError(f"Expected label and pixels columns in {csv_path}; columns={list(df.columns)}")

    counts = df[label_col].value_counts().sort_index().to_dict()
    report = {
        "csv_path": str(csv_path),
        "num_rows": int(len(df)),
        "label_column": str(label_col),
        "pixel_column": str(pixel_col),
        "counts": {CKPLUS_LABELS.get(int(k), str(k)): int(v) for k, v in counts.items()},
    }

    for label_id, label_name in CKPLUS_LABELS.items():
        subset = df[df[label_col] == label_id].head(num_per_class)
        samples = []
        for row_idx, row in subset.iterrows():
            image = read_pixels(row[pixel_col], image_size)
            samples.append((image, f"row {row_idx}"))
        make_contact_sheet(samples, f"CK+ label {label_id}: {label_name}", output_dir / f"ckplus_{label_id}_{label_name}.png")

    return report


def main():
    parser = argparse.ArgumentParser(description="Create visual label-semantic checks for RAF-DB and CK+.")
    parser.add_argument("--raf_path", default=None, help="Path to RAF-compatible dataset root")
    parser.add_argument("--ck_path", default=None, help="Path to CK+ dataset root containing ckextended.csv")
    parser.add_argument("--output_dir", default="label_semantics_check")
    parser.add_argument("--num_per_class", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=112)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    if args.raf_path:
        report["raf"] = verify_raf(args.raf_path, output_dir, args.num_per_class, args.image_size)
    if args.ck_path:
        report["ckplus"] = verify_ckplus(args.ck_path, output_dir, args.num_per_class, args.image_size)

    report_path = output_dir / "label_semantics_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote visual checks to {output_dir}")


if __name__ == "__main__":
    main()
