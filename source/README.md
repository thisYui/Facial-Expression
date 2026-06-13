# Repository Modifications & Additions

This document details all the modifications, improvements, and new files added to this repository compared to the original author's codebase at [zyh-uaiaaaa/Generalizable-FER](https://github.com/zyh-uaiaaaa/Generalizable-FER). 

While the original codebase only provided the core training logic on RAF-DB (`ours_CAFE.py`) and the CLIP model wrapper, this repository has been extended with a robust data-preparation pipeline, zero-shot cross-dataset evaluation scripts, label verification checks, and structured logs.

---

## 1. Directory Structure of Additions & Modifications

Below is the layout of the files added or modified in this project:
```text
code/
├── requirements.txt (NEW: Environment dependencies definition)
├── notebooks
│   └── training.ipynb (NEW: Notebook for training & cross-dataset evaluation)
├── code/
│   ├── evaluate_ckplus48.py (NEW: Customized CK+48 dataset evaluator)
│   └── evaluate_cross_dataset.py (NEW: Zero-shot cross-dataset evaluator)
├── scripts/ (NEW: Data preparation & validation tools)
│   ├── create_raf_compatible.py
│   ├── verify_label_semantics.py
│   └── clean_results.py
├── cross_dataset_evaluations/ (NEW: Zero-shot evaluation outputs)
│   ├── ckplus48_eval/
│   ├── ckplus48_peak_eval/
│   ├── ferplus_eval/
│   ├── affectnet_eval/
│   ├── sfew_val_eval/
│   ├── mma_eval/
│   └── cross_dataset_summary.csv
└── result/ (NEW: Checkpoints and training logs on RAF-DB)
    ├── ours_best.pth (Best model checkpoint on RAF-DB)
    ├── ours_final.pth
    ├── results.txt (Raw training accuracy logs)
    └── results_clean.csv (Cleaned tabular training logs)
```

---

## 2. Details of New Components

### 2.1. Evaluation Scripts
*   **[code/evaluate_ckplus48.py](code/evaluate_ckplus48.py)**:
    *   **Objective**: Evaluate the model (trained on RAF-DB) on the CK+48 dataset.
    *   **Key Features**:
        *   Supports loading from either subdirectories (class names) or a CSV file (`ckextended.csv`) containing flattened raw pixel vectors.
        *   Maps CK+48 emotions to the CAFE 7-class schema, dropping the unsupported `contempt` category.
        *   Provides an `--exclude_neutral` flag to run a **Peak-only** 6-class evaluation.
        *   Standardizes preprocessing: Handles grayscale-to-RGB conversion, resizes images from 48x48 to 224x224, and normalizes them using ImageNet statistics.
        *   Saves visual outputs: Class distribution bar chart (`class_distribution.png`), confusion matrix heatmap (`confusion_matrix.png`), metric statistics (`metrics.json`), and sample-level predictions (`predictions.csv`).
*   **[code/evaluate_cross_dataset.py](code/evaluate_cross_dataset.py)**:
    *   **Objective**: Run zero-shot cross-dataset evaluations on standard benchmarks mentioned in the paper.
    *   **Supported Datasets**: **FERPlus**, **AffectNet**, **SFEW 2.0 (Val)**, and **MMAFEDB**.
    *   **Key Features**:
        *   Resolves directory discrepancies across Kaggle dataset exports (e.g., AffectNet's nesting under `archive (3)`, SFEW's lack of public labels in `Test` mapped to using `Val`).
        *   Normalizes folder-based classifications (both numeric and text-based) into CAFE's 7-class target.
        *   Excludes the `contempt` category, tracking skipped sample counts per class.
        *   Saves evaluation logs, confusion matrices, and detailed predictions.

### 2.2. Preprocessing & Verification Scripts
*   **[scripts/create_raf_compatible.py](scripts/create_raf_compatible.py)**:
    *   **Objective**: Convert custom RAF-DB train/test directories and CSV labels into the official format expected by the original CAFE code (`EmoLabel/list_patition_label.txt` and `Image/aligned/`).
    *   Uses symbolic links (symlinks) to save disk space and accelerate setups on Kaggle.
*   **[scripts/verify_label_semantics.py](scripts/verify_label_semantics.py)**:
    *   **Objective**: Visually verify mapped labels against actual images to ensure label semantics are accurate.
    *   Draws a grid of sample images (contact sheet) per class so the researcher can quickly double-check that expression classes match the facial features.
*   **[scripts/clean_results.py](scripts/clean_results.py)**:
    *   **Objective**: Parse the original unstructured `results.txt` output from the training process, clean up PyTorch tensor formats, and write a structured table to `results_clean.csv`, outputting the best epoch and accuracy.

### 2.3. Configuration & Execution Notebooks
*   **[requirements.txt](requirements.txt)**:
    *   Lists all necessary Python dependencies to ensure reproducible runtime environments (including PyTorch, OpenCV, CLIP dependencies, scikit-learn, and plotting tools).
*   **[PLAN.md](PLAN.md)**:
    *   A comprehensive implementation plan outlining path conventions, GPU optimizations, batch size configs, and evaluation steps for zero-shot experiments.
*   **[train_notebook.ipynb](train_notebook.ipynb)**:
    *   The execution notebook for training the CAFE model on RAF-DB (for 60 epochs), checkpointing the best weights, and automatically evaluating the cross-dataset benchmarks.

---

## 3. Experimental Results Summary

The table below compares the original paper's reported accuracy against our reproduced results (trained on RAF-DB and evaluated zero-shot across other datasets):

| Evaluation Dataset (Test Set) | Paper Accuracy (%) | Reproduced Accuracy (%) | Configuration / Evaluation Notes |
| :--- | :---: | :---: | :--- |
| **RAF-DB** | 88.72% | **89.15%** | Training source dataset (Outperformed the paper) |
| **CK+48 (7 classes)** | - | **82.48%** | Zero-shot evaluation including Neutral class |
| **CK+48 (6 classes)** | - | **64.72%** | Zero-shot peak-only evaluation (no Neutral/Contempt) |
| **FERPlus** | 73.16% | **66.81%** | Zero-shot evaluation; Contempt class skipped |
| **AffectNet** | 45.86% | **52.01%** | Zero-shot evaluation (Outperformed the paper) |
| **SFEW 2.0** | 52.86% | **46.87%** | Zero-shot evaluation on the Validation split |
| **MMAFEDB** | 56.80% | **56.72%** | Zero-shot evaluation; comparable to paper |

*Detailed prediction logs, metric outputs, and confusion matrix charts are located in the [cross_dataset_evaluations/](cross_dataset_evaluations) directory.*

---

## 4. Usage Commands Quick-Start

### 1. Data Preparation
Convert a custom RAF-DB setup to the CAFE-compatible layout:
```bash
python scripts/create_raf_compatible.py --src /path/to/raw/raf-db --dst /path/to/raf-db-compatible
```

### 2. Verify Label Mappings
Generate visual contact sheets to inspect label alignment:
```bash
python scripts/verify_label_semantics.py --raf_path /path/to/raf-db-compatible --ck_path /path/to/ckplus48 --output_dir label_semantics_check
```

### 3. Evaluate CK+48
Run evaluations on CK+48 (using either folders or `ckextended.csv`):
```bash
# 7-Class (with Neutral)
python code/evaluate_ckplus48.py --ck_path /path/to/ckplus48 --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/ckplus48_eval

# 6-Class (Peak-only)
python code/evaluate_ckplus48.py --ck_path /path/to/ckplus48 --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/ckplus48_peak_eval --exclude_neutral
```

### 4. Evaluate Paper Datasets
Run zero-shot tests on standard paper splits:
```bash
python code/evaluate_cross_dataset.py --dataset ferplus --data_path /path/to/ferplus --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/ferplus_eval
python code/evaluate_cross_dataset.py --dataset affectnet --data_path /path/to/affectnet --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/affectnet_eval
python code/evaluate_cross_dataset.py --dataset sfew --data_path /path/to/sfew --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/sfew_val_eval
python code/evaluate_cross_dataset.py --dataset mma --data_path /path/to/mma --checkpoint result/ours_best.pth --output_dir cross_dataset_evaluations/mma_eval
```

### 5. Parse Logs
Convert training logs to CSV:
```bash
python scripts/clean_results.py result/results.txt
```
