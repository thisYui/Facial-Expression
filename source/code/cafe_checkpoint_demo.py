import argparse
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
import clip  # noqa: E402


CAFE_CLASS_NAMES = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sadness",
    5: "anger",
    6: "neutral",
}


def select_device(force_cpu: bool) -> torch.device:
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.relu(x + identity)
        return x


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
        self.in_channels = channels[0]
        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block=BasicBlock, n_blocks=2, channels=64, stride=1):
        layers = []
        downsample = self.in_channels != block.expansion * channels
        layers.append(block(self.in_channels, channels, stride, downsample))
        for _ in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x, h


class Model(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0):
        super().__init__()
        res18 = ResNet(
            block=BasicBlock,
            n_blocks=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            output_dim=1000,
        )
        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])
        fc_in_dim = list(res18.children())[-1].in_features
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x, clip_model, targets=None, phase="test"):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)

        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        return out, out


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class Prediction:
    pred_idx: int
    confidence: float
    box: tuple[int, int, int, int]


def load_model(checkpoint_path, device):
    model = Model(num_classes=7)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    return model, clip_model


def load_face_detector():
    detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(detector_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load face detector: {detector_path}")
    return detector


def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def crop_with_margin(frame, x, y, w, h, margin=0.25):
    h_img, w_img = frame.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


@torch.inference_mode()
def predict_face(model, clip_model, frame_bgr, face, device, face_margin=0.25):
    if face is None:
        crop = frame_bgr
        box = (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])
    else:
        crop, box = crop_with_margin(frame_bgr, *face, margin=face_margin)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    input_tensor = (input_tensor - IMAGE_MEAN) / IMAGE_STD
    input_tensor = input_tensor.unsqueeze(0).to(device)
    logits, _ = model(input_tensor, clip_model, phase="test")
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    return pred_idx, confidence, box


def detect_faces(detector, frame, min_face, detect_width):
    if detect_width > 0 and frame.shape[1] > detect_width:
        scale = detect_width / frame.shape[1]
        detect_frame = cv2.resize(
            frame,
            (detect_width, int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        detect_frame = frame

    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
    scaled_min_face = max(20, int(min_face * scale))
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(scaled_min_face, scaled_min_face),
    )

    if scale == 1.0 or len(faces) == 0:
        return faces

    inv_scale = 1.0 / scale
    return np.array(
        [
            (
                int(x * inv_scale),
                int(y * inv_scale),
                int(w * inv_scale),
                int(h * inv_scale),
            )
            for x, y, w, h in faces
        ],
        dtype=np.int32,
    )


def draw_overlay(frame, label, confidence, box):
    x1, y1, x2, y2 = box
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    y_text = max(30, y1 - 10)
    cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam demo for the CAFE FER checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "result" / "ours_best.pth"),
        help="Path to the checkpoint .pth file.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode even if CUDA is available.")
    parser.add_argument("--face-margin", type=float, default=0.25, help="Extra margin around detected face.")
    parser.add_argument("--min-face", type=int, default=120, help="Minimum face size in pixels.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=360, help="Camera capture height.")
    parser.add_argument("--display-width", type=int, default=1280, help="Display window width.")
    parser.add_argument("--display-height", type=int, default=720, help="Display window height.")
    parser.add_argument(
        "--detect-width",
        type=int,
        default=320,
        help="Downscaled frame width for face detection. Use 0 to detect at capture size.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=3,
        help="Run emotion inference every N frames and reuse the last result between runs.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Maximum faces to classify per frame. The largest faces are used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.cpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, clip_model = load_model(args.checkpoint, device)
    detector = load_face_detector()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "CAFE FER webcam demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.display_width, args.display_height)

    fps_t0 = time.time()
    fps_counter = 0
    fps_value = 0.0
    frame_idx = 0
    predictions: list[Prediction] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame_idx += 1
            faces = detect_faces(detector, frame, args.min_face, args.detect_width)
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[: args.max_faces]

            should_infer = bool(faces) and (
                not predictions or frame_idx % max(1, args.infer_every) == 0
            )
            if should_infer:
                predictions = []
                for face in faces:
                    pred_idx, confidence, box = predict_face(
                        model, clip_model, frame, face, device, face_margin=args.face_margin
                    )
                    predictions.append(Prediction(pred_idx, confidence, box))
            else:
                for i, face in enumerate(faces[: len(predictions)]):
                    _, box = crop_with_margin(frame, *face, margin=args.face_margin)
                    predictions[i].box = box

            if not faces:
                predictions = []

            for prediction in predictions:
                label = CAFE_CLASS_NAMES[prediction.pred_idx]
                draw_overlay(frame, label, prediction.confidence, prediction.box)

            fps_counter += 1
            now = time.time()
            elapsed = now - fps_t0
            if elapsed >= 1.0:
                fps_value = fps_counter / elapsed
                fps_t0 = now
                fps_counter = 0
            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
