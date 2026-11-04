#!/usr/bin/env python3
"""
Turn an existing sign language image dataset into model/landmarks.json for GestureX.

Expected folder layout (this is how the Kaggle "ASL Alphabet" dataset is organised):

    dataset/
      A/  img1.jpg img2.jpg ...
      B/  ...
      hello/ ...

Each sub-folder name becomes the sign's label. For every image, MediaPipe finds the
21 hand points, which are normalised exactly as index.html does it, and saved.

    pip install mediapipe opencv-python
    python build_dataset.py "C:\\path\\to\\dataset" --per-class 300

The first run downloads the small hand model (about 8 MB) into model/hand_landmarker.task,
so it needs an internet connection. Then reload the app; the Dataset panel reports
how many samples were loaded.
"""
import argparse, json, math, os, random, sys, urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def normalise(lm, handed):
    # must match normalise() in index.html
    w = lm[0]
    m = -1 if handed == "Left" else 1
    pts = [((p.x - w.x) * m, p.y - w.y, p.z - w.z) for p in lm]
    s = max(math.hypot(p[0], p[1]) for p in pts) or 1
    return [round(c / s, 4) for p in pts for c in p]


def ensure_model(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder that contains one sub-folder per sign")
    ap.add_argument("--per-class", type=int, default=300, help="max images per label")
    ap.add_argument("--out", default="model/landmarks.json")
    ap.add_argument("--model", default="model/hand_landmarker.task")
    ap.add_argument("--flip", action="store_true",
                    help="mirror every image first (use if your dataset photos are selfie-mirrored)")
    a = ap.parse_args()

    ensure_model(a.model)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    samples, skipped = [], 0
    labels = sorted(d for d in os.listdir(a.folder) if os.path.isdir(os.path.join(a.folder, d)))
    if not labels:
        sys.exit("No label sub-folders found in " + a.folder)

    for label in labels:
        d = os.path.join(a.folder, label)
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        got = 0
        for f in files:
            if got >= a.per_class:
                break
            img = cv2.imread(os.path.join(d, f))
            if img is None:
                continue
            if a.flip:
                img = cv2.flip(img, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                skipped += 1
                continue
            handed = res.handedness[0][0].category_name
            samples.append({"l": label, "v": normalise(res.hand_landmarks[0], handed)})
            got += 1
        print(f"{label}: {got} samples")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"samples": samples}, fh, separators=(",", ":"))
    print(f"Wrote {len(samples)} samples to {a.out} ({skipped} images had no hand)")


if __name__ == "__main__":
    main()