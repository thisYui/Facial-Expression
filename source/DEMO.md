# Facial Expression Recognition Demo Guide

This guide explains how to run the webcam demos in this project. Run all commands from the repository root

## 1. Environment Setup

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the main runtime dependencies:

```bash
python -c "import torch, cv2; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('opencv:', cv2.__version__)"
```

Notes:

- A working webcam is required.
- The CAFE checkpoint demo uses `code/ours_best.pth` by default.
- The first CLIP run may take extra time while loading the `ViT-B/32` model.

## 2. CAFE Checkpoint Demo

Demo file:

```text
code/cafe_checkpoint_demo.py
```

This demo uses the trained CAFE model to classify seven facial expressions:

```text
surprise, fear, disgust, happy, sadness, anger, neutral
```

Run with the default settings:

```bash
python code/cafe_checkpoint_demo.py
```

Run with an explicit checkpoint:

```bash
python code/cafe_checkpoint_demo.py --checkpoint code/ours_best.pth
```

Common options:

| Option | Description |
|---|---|
| `--checkpoint` | Path to the CAFE `.pth` checkpoint |
| `--camera-index` | Webcam index, default is `0` |
| `--cpu` | Force CPU inference |
| `--min-face` | Minimum detected face size |
| `--face-margin` | Extra margin around the detected face before prediction |
| `--infer-every` | Run inference every N frames |
| `--max-faces` | Maximum number of faces to classify per frame |

## 3. CLIP Zero-Shot Demo

Demo file:

```text
code/clip_zero_shot_demo.py
```

This demo does not require a checkpoint. It uses CLIP zero-shot classification with text prompts to predict facial expressions.

Run with the default settings:

```bash
python code/clip_zero_shot_demo.py
```

Main differences:

| Demo | Requires checkpoint | Expected accuracy | Purpose |
|---|---:|---|---|
| `cafe_checkpoint_demo.py` | Yes | Higher | Demo the trained model |
| `clip_zero_shot_demo.py` | No | Lower | Quick CLIP zero-shot demo |

## 4. Demo Controls

- Press `q` to quit.
- Press `ESC` to quit.
- Closing the OpenCV window with the `X` button also stops the demo.

## 5. Troubleshooting

| Issue | Fix |
|---|---|
| Webcam cannot be opened | Try `--camera-index 1` or `--camera 1` |
| Black screen or no frames | Check webcam permissions and make sure no other app is using it |
| CUDA out of memory | Add `--cpu` or reduce the capture resolution |
| Low FPS | Reduce `--width` and `--height`, or increase `--infer-every` for the CAFE demo |
| Checkpoint not found | Use `--checkpoint code/ours_best.pth` or an absolute path |
