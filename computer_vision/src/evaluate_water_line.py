
import cv2
import numpy as np

from ultralytics import YOLO
from estimate_water_level import (
    MODEL_PATH,
    IMAGES_DIR,
    estimate_water_line,
)

LABELS_DIR = IMAGES_DIR.parent / "labels"


def load_annotations(path, width, height):
    annotations = []

    for line in path.read_text().splitlines():
        values = list(map(float, line.split()))
        class_id = int(values[0])
        xc, yc, w, h = values[1:]

        annotations.append({
            "class_id": class_id,
            "xc": xc * width,
            "yc": yc * height,
            "box": [
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            ],
        })

    return annotations


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def main():
    model = YOLO(str(MODEL_PATH))

    errors_px = []
    errors_normalized = []

    total_gauges = 0
    matched_gauges = 0
    missing_water_annotations = 0
    failed_estimates = 0

    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        annotations = load_annotations(
            label_path, width, height
        )

        gauges = [
            a for a in annotations
            if a["class_id"] == 0
        ]

        waters = [
            a for a in annotations
            if a["class_id"] == 1
        ]

        total_gauges += len(gauges)

        results = model.predict(
            image,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False,
        )

        predicted_boxes = [
            box.xyxy[0].cpu().numpy()
            for box in results[0].boxes
            if int(box.cls[0]) == 0
        ]

        # Correspondências por maior IoU, sem reutilização
        candidates = []

        for gi, gauge in enumerate(gauges):
            for pi, predicted in enumerate(predicted_boxes):
                iou = calculate_iou(gauge["box"], predicted)

                if iou >= 0.5:
                    candidates.append((iou, gi, pi))

        candidates.sort(reverse=True)

        used_gauges = set()
        used_predictions = set()

        for iou, gi, pi in candidates:
            if gi in used_gauges or pi in used_predictions:
                continue

            used_gauges.add(gi)
            used_predictions.add(pi)
            matched_gauges += 1

            gauge = gauges[gi]

            if not waters:
                missing_water_annotations += 1
                continue

            # Associação exploratória por proximidade horizontal
            water = min(
                waters,
                key=lambda w: abs(w["xc"] - gauge["xc"])
            )

            estimated_y = estimate_water_line(
                image, predicted_boxes[pi]
            )

            if estimated_y is None:
                failed_estimates += 1
                continue

            error_px = abs(estimated_y - water["yc"])

            errors_px.append(error_px)
            errors_normalized.append(
                100 * error_px / height
            )

    print("\n===== AVALIAÇÃO CORRIGIDA =====")
    print("Réguas reais:", total_gauges)
    print("Réguas associadas (IoU >= 0.5):", matched_gauges)
    print(
        "Réguas sem correspondência:",
        total_gauges - matched_gauges
    )
    print(
        "Sem anotação de água:",
        missing_water_annotations
    )
    print(
        "Falhas na estimativa:",
        failed_estimates
    )
    print("Estimativas avaliadas:", len(errors_px))

    if errors_px:
        print(
            f"Erro médio: {np.mean(errors_px):.2f} px"
        )
        print(
            f"Erro mediano: {np.median(errors_px):.2f} px"
        )
        print(
            f"Erro médio normalizado: "
            f"{np.mean(errors_normalized):.2f}%"
        )
        print(
            f"Erro mediano normalizado: "
            f"{np.median(errors_normalized):.2f}%"
        )


if __name__ == "__main__":
    main()
