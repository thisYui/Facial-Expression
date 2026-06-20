import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import clip


EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "surprised",
    "fearful",
    "disgusted",
    "angry",
]


def build_text_prompts() -> List[str]:
    # Multiple prompt templates usually improve zero-shot robustness.
    templates = [
        "a face with a {} expression",
        "a photo of a person who looks {}",
        "a portrait showing {} emotion",
    ]
    prompts = []
    for emotion in EMOTIONS:
        prompts.extend([t.format(emotion) for t in templates])
    return prompts


def select_device(force_cpu: bool) -> torch.device:
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip_and_text_features(device: torch.device) -> Tuple[torch.nn.Module, torch.Tensor, object]:
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    prompts = build_text_prompts()
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, text_features, preprocess


def aggregate_prompt_scores(prompt_probs: np.ndarray) -> np.ndarray:
    # prompt_probs shape: [len(EMOTIONS) * n_templates]
    n_templates = prompt_probs.shape[0] // len(EMOTIONS)
    scores = prompt_probs.reshape(len(EMOTIONS), n_templates).mean(axis=1)
    return scores


def predict_emotion(
    bgr_face: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[str, float]:
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().numpy()

    emotion_scores = aggregate_prompt_scores(probs)
    idx = int(np.argmax(emotion_scores))
    conf = float(emotion_scores[idx])
    return EMOTIONS[idx], conf


def main():
    parser = argparse.ArgumentParser(description="Realtime facial emotion demo with CLIP zero-shot")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--min-face", type=int, default=120, help="Minimum face size in pixels")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    device = select_device(args.cpu)
    model, text_features, preprocess = load_clip_and_text_features(device)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV Haar face detector.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    prev_t = time.time()

    print("Press 'q' to quit.")
    # Create a resizable window and set to 16:9 (1280x720) resolution.
    window_name = "Realtime Emotion Demo (CLIP zero-shot)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(args.min_face, args.min_face),
        )

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            if face.size == 0:
                continue

            label, conf = predict_emotion(face, model, preprocess, text_features, device)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 220, 40),
                2,
                cv2.LINE_AA,
            )

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Device: {device}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        # Quit on 'q' or ESC (27)
        if key == ord("q") or key == 27:
            break

        # If the user closed the window using the X button, exit.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
