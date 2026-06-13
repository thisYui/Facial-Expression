import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(r"^\s*(\d+)_tensor\(([-+0-9.eE]+),")


def parse_results(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        match = RESULT_RE.match(line)
        if not match:
            raise ValueError(f"Cannot parse line {line_no}: {line}")
        epoch = int(match.group(1))
        accuracy = float(match.group(2))
        rows.append((epoch, accuracy, accuracy * 100.0))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert CAFE results.txt tensor logs to CSV.")
    parser.add_argument("input", help="Path to original results.txt")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("results_clean.csv")

    rows = parse_results(input_path)
    best = max(rows, key=lambda row: row[1])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "accuracy_percent"])
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    print(f"Best epoch: {best[0]}")
    print(f"Best accuracy: {best[1]:.4f} ({best[2]:.2f}%)")


if __name__ == "__main__":
    main()
