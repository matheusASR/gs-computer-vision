
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR
from evaluate_water_line import load_annotations, calculate_iou

LABELS_DIR = IMAGES_DIR.parent / "labels"

CONFIDENCE = 0.10
GAUGE_IOU = 0.50
WATER_IOU = 0.10


def center_x(box):
    return (box[0] + box[2]) / 2


def center_y(box):
    return (box[1] + box[3]) / 2


def match_boxes(ground_truth, predictions, threshold):
    """Associa caixas por IoU, sem reutilizar objetos."""
    candidates = []

    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predictions):
            iou = calculate_iou(gt["box"], pred)
            if iou >= threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi))

    return matches


def evaluate(confidence, model):
    total_gauges = 0
    matched_gauges = 0
    total_waters = 0
    matched_waters = 0

    water_predictions = 0
    errors = []

    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        annotations = load_annotations(label_path, width, height)

        gauges = [a for a in annotations if a["class_id"] == 0]
        waters = [a for a in annotations if a["class_id"] == 1]

        total_gauges += len(gauges)
        total_waters += len(waters)

        result = model.predict(
            source=image,
            imgsz=416,
            conf=confidence,
            device="cpu",
            verbose=False,
        )[0]

        pred_gauges = []
        pred_waters = []

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])

            if class_id == 0:
                pred_gauges.append(coords)
            elif class_id == 1:
                pred_waters.append(coords)

        water_predictions += len(pred_waters)

        gauge_matches = match_boxes(
            gauges, pred_gauges, GAUGE_IOU
        )
        water_matches = match_boxes(
            waters, pred_waters, WATER_IOU
        )

        matched_gauges += len(gauge_matches)
        matched_waters += len(water_matches)

        for gi, pi in water_matches:
            true_y = waters[gi]["yc"]
            predicted_y = center_y(pred_waters[pi])

            error = 100 * abs(predicted_y - true_y) / height
            errors.append(error)

    print(f"\n===== CONFIANÇA {confidence:.2f} =====")
    print(f"Réguas reais: {total_gauges}")
    print(f"Réguas detectadas (IoU >= 0.5): {matched_gauges}")
    print(f"Águas reais: {total_waters}")
    print(f"Águas detectadas (IoU >= 0.1): {matched_waters}")
    print(f"Total de previsões de água: {water_predictions}")

    if total_waters:
        print(
            f"Recall da água (IoU >= 0.1): "
            f"{100 * matched_waters / total_waters:.2f}%"
        )

    if water_predictions:
        print(
            f"Precisão aproximada da água (IoU >= 0.1): "
            f"{100 * matched_waters / water_predictions:.2f}%"
        )

    if errors:
        print(
            f"Erro médio vertical: {np.mean(errors):.2f}%"
        )
        print(
            f"Erro mediano vertical: {np.median(errors):.2f}%"
        )



def main():
    model = YOLO(str(MODEL_PATH))

    for confidence in [0.10, 0.15, 0.20, 0.25]:
        evaluate(confidence, model)


if __name__ == "__main__":
    main()


