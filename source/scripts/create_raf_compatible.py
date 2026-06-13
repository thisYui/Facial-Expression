import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def find_col(df, candidates, fallback_index):
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return df.columns[fallback_index]


def normalize_labels(series):
    numeric = pd.to_numeric(series)
    unique = sorted(int(value) for value in numeric.unique())
    min_label = min(unique)
    max_label = max(unique)

    if min_label == 1 and max_label == 7:
        return numeric.astype(int), "kept_1_to_7"
    if min_label == 0 and max_label == 6:
        return (numeric.astype(int) + 1), "converted_0_to_6_into_1_to_7"

    raise ValueError(
        f"Unsupported RAF label range {unique}. Expected 1..7 or 0..6."
    )


def build_split(src_root, dst_image_dir, csv_name, split_name):
    csv_path = src_root / csv_name
    df = pd.read_csv(csv_path)

    image_col = find_col(df, ["image", "image_name", "filename", "file", "path", "img"], 0)
    label_col = find_col(df, ["label", "emotion", "class", "target"], 1)
    labels, label_mode = normalize_labels(df[label_col])

    split_dir = src_root / "DATASET" / split_name
    all_images = list(split_dir.rglob("*.*"))
    by_name = {path.name: path for path in all_images}
    by_stem = {path.stem: path for path in all_images}

    rows = []
    missing = []

    for index, row in df.iterrows():
        raw_image = str(row[image_col])
        candidate = Path(raw_image)
        source = None

        if candidate.is_absolute() and candidate.exists():
            source = candidate
        else:
            rel = split_dir / raw_image
            if rel.exists():
                source = rel
            elif Path(raw_image).name in by_name:
                source = by_name[Path(raw_image).name]
            elif Path(raw_image).stem in by_stem:
                source = by_stem[Path(raw_image).stem]

        if source is None:
            missing.append(raw_image)
            continue

        base_name = f"{split_name}_{index:05d}"
        label_file_name = f"{base_name}.jpg"
        target = dst_image_dir / f"{base_name}_aligned.jpg"
        os.symlink(source, target)
        rows.append(f"{label_file_name} {int(labels.iloc[index])}")

    return {
        "csv": str(csv_path),
        "split": split_name,
        "rows": rows,
        "num_rows": len(rows),
        "label_mode": label_mode,
        "missing": missing,
        "label_counts": labels.value_counts().sort_index().astype(int).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Create RAF-DB layout expected by official CAFE code.")
    parser.add_argument("--src", default="/kaggle/working/data/raf-basic")
    parser.add_argument("--dst", default="/kaggle/working/data/raf-basic-compatible")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    emo_dir = dst_root / "EmoLabel"
    image_dir = dst_root / "Image" / "aligned"

    if dst_root.exists():
        shutil.rmtree(dst_root)
    emo_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    train = build_split(src_root, image_dir, "train_labels.csv", "train")
    test = build_split(src_root, image_dir, "test_labels.csv", "test")

    all_rows = train["rows"] + test["rows"]
    (emo_dir / "list_patition_label.txt").write_text("\n".join(all_rows) + "\n", encoding="utf-8")

    print("Created", dst_root)
    print("Train:", train["num_rows"], train["label_mode"], train["label_counts"])
    print("Test:", test["num_rows"], test["label_mode"], test["label_counts"])
    if train["missing"] or test["missing"]:
        print("Missing images:", len(train["missing"]) + len(test["missing"]))


if __name__ == "__main__":
    main()
