import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/hamsab/projects/altitude-aware-yolo")
RUN = "baseline-paper-10ep"
WEIGHTS = "best"
SPLIT = "test"

PRED_JSON = ROOT / "results" / "predictions" / f"{RUN}_{WEIGHTS}_{SPLIT}.json"
IMG_DIR = ROOT / "data" / "processed" / "images" / SPLIT
LBL_DIR = ROOT / "data" / "processed" / "labels" / SPLIT
OUT_DIR = ROOT / "results" / "prediction_samples" / RUN
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 50

print("Pred JSON:", PRED_JSON)
print("Images:   ", IMG_DIR)
print("Labels:   ", LBL_DIR)
print("Out:      ", OUT_DIR)

preds = json.loads(PRED_JSON.read_text())

def draw_poly(img, pts, color, thickness=2):
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_gt(img, stem):
    h, w = img.shape[:2]
    label_path = LBL_DIR / f"{stem}.txt"
    if not label_path.exists():
        return

    for line in label_path.read_text().splitlines():
        vals = line.strip().split()
        if len(vals) != 9:
            continue

        coords = np.array([float(x) for x in vals[1:]], dtype=float).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        draw_poly(img, coords, (0, 255, 0), 2)

def normalize_pred_boxes(pdata):
    """
    Supports common JSON shapes:
      {"boxes": [{"corners": [...], "conf": ...}, ...]}
      {"predictions": [...]}
      [{"corners": [...], "conf": ...}, ...]
    """
    if isinstance(pdata, dict):
        boxes = pdata.get("boxes") or pdata.get("predictions") or pdata.get("objects") or []
    elif isinstance(pdata, list):
        boxes = pdata
    else:
        boxes = []
    return boxes

def extract_pts_and_conf(box):
    if isinstance(box, dict):
        pts = (
            box.get("corners")
            or box.get("points")
            or box.get("xyxyxyxy")
            or box.get("obb")
        )
        conf = box.get("conf", box.get("confidence", box.get("score")))
    else:
        pts = box
        conf = None

    if pts is None:
        return None, conf

    arr = np.asarray(pts, dtype=float)
    if arr.size != 8:
        return None, conf

    return arr.reshape(4, 2), conf

saved = 0

for key, pdata in preds.items():
    stem = Path(key).stem
    img_path = IMG_DIR / f"{stem}.jpg"

    if not img_path.exists():
        print("missing image:", img_path)
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print("could not read:", img_path)
        continue

    draw_gt(img, stem)

    for box in normalize_pred_boxes(pdata):
        pts, conf = extract_pts_and_conf(box)
        if pts is None:
            continue

        draw_poly(img, pts, (0, 0, 255), 2)

        if conf is not None:
            x, y = pts[0].astype(int)
            cv2.putText(
                img,
                f"{float(conf):.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    out_path = OUT_DIR / f"{saved + 1:03d}_{stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print("saved:", out_path)
    saved += 1

    if saved >= MAX_IMAGES:
        break

print(f"Done. Saved {saved} images.")
