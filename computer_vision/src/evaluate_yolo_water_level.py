
import cv2
import numpy as np
from ultralytics import YOLO

from estimate_water_level import (
    MODEL_PATH,
    IMAGES_DIR,
    estimate_water_line,
)

from evaluate_water_line import (
    load_annotations,
    calculate_iou,
)

LABELS_DIR = IMAGES_DIR.parent / "labels"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


def center_x(box):
    return (box[0] + box[2]) / 2


def center_y(box):
    return (box[1] + box[3]) / 2


def match_gauges(gauges, predictions):
    """Associa réguas reais e previstas sem reutilização."""
    candidates = []

    for gi, gauge in enumerate(gauges):
        for pi, predicted in enumerate(predictions):
            iou = calculate_iou(gauge["box"], predicted)

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matches = []
    used_gauges = set()
    used_predictions = set()

    for iou, gi, pi in candidates:
        if gi in used_gauges or pi in used_predictions:
            continue

        used_gauges.add(gi)
        used_predictions.add(pi)
        matches.append((gi, pi))

    return matches


def main():
    model = YOLO(str(MODEL_PATH))

    opencv_errors = []
    yolo_errors = []
    paired_opencv_errors = []
    paired_yolo_errors = []

    total_gauges = 0
    matched_gauges = 0
    eligible_pairs = 0
    yolo_detected = 0

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

        result = model.predict(
            source=image,
            imgsz=416,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
        )[0]

        predicted_gauges = []
        predicted_waters = []

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()

            if int(box.cls[0]) == 0:
                predicted_gauges.append(coords)

            elif int(box.cls[0]) == 1:
                predicted_waters.append(coords)

        matches = match_gauges(
            gauges, predicted_gauges
        )

        matched_gauges += len(matches)

        # Evita reutilizar a mesma anotação de água
        # em diferentes réguas reais.
        used_water_labels = set()

        for gi, pi in matches:
            gauge = gauges[gi]
            predicted_gauge = predicted_gauges[pi]

            available_waters = [
                (wi, water)
                for wi, water in enumerate(waters)
                if wi not in used_water_labels
            ]

            if not available_waters:
                continue

            # Associação exploratória pela distância horizontal.
            wi, true_water = min(
                available_waters,
                key=lambda item: abs(
                    item[1]["xc"] - gauge["xc"]
                )
            )

            used_water_labels.add(wi)
            eligible_pairs += 1

            true_y = true_water["yc"]

            # Estratégia A: OpenCV
            opencv_y = estimate_water_line(
                image, predicted_gauge
            )

            opencv_error = None

            if opencv_y is not None:
                opencv_error = (
                    100 * abs(opencv_y - true_y) / height
                )
                opencv_errors.append(opencv_error)

            # Estratégia B: YOLO
            # Busca água próxima da régua prevista.
            gauge_width = max(
                1, predicted_gauge[2] - predicted_gauge[0]
            )

            candidates = [
                water
                for water in predicted_waters
                if abs(
                    center_x(water) -
                    center_x(predicted_gauge)
                ) <= gauge_width
            ]

            if not candidates:
                continue

            # Escolhe a detecção mais próxima da base da régua.
            predicted_water = min(
                candidates,
                key=lambda water: abs(
                    center_y(water) - predicted_gauge[3]
                )
            )

            predicted_y = center_y(predicted_water)

            yolo_error = (
                100 * abs(predicted_y - true_y) / height
            )

            yolo_errors.append(yolo_error)
            yolo_detected += 1

            if opencv_error is not None:
                paired_opencv_errors.append(opencv_error)
                paired_yolo_errors.append(yolo_error)

    print("\n===== COMPARAÇÃO YOLO VS OPENCV =====")
    print("Réguas reais:", total_gauges)
    print("Réguas associadas:", matched_gauges)
    print("Pares elegíveis:", eligible_pairs)

    print("\n===== OPENCV =====")
    print("Estimativas:", len(opencv_errors))

    if opencv_errors:
        print(
            f"Erro médio normalizado: "
            f"{np.mean(opencv_errors):.2f}%"
        )
        print(
            f"Erro mediano normalizado: "
            f"{np.median(opencv_errors):.2f}%"
        )

    print("\n===== YOLO WATER_LEVEL =====")
    print("Estimativas:", yolo_detected)
    print(
        "Sem detecção de água:",
        eligible_pairs - yolo_detected
    )

    if yolo_errors:
        print(
            f"Erro médio normalizado: "
            f"{np.mean(yolo_errors):.2f}%"
        )
        print(
            f"Erro mediano normalizado: "
            f"{np.median(yolo_errors):.2f}%"
        )

    print("\n===== COMPARAÇÃO NOS MESMOS CASOS =====")
    print("Pares comparáveis:", len(paired_yolo_errors))

    if paired_yolo_errors:
        print(
            f"OpenCV - erro médio: "
            f"{np.mean(paired_opencv_errors):.2f}%"
        )
        print(
            f"YOLO - erro médio: "
            f"{np.mean(paired_yolo_errors):.2f}%"
        )
        print(
            f"OpenCV - erro mediano: "
            f"{np.median(paired_opencv_errors):.2f}%"
        )
        print(
            f"YOLO - erro mediano: "
            f"{np.median(paired_yolo_errors):.2f}%"
        )


if __name__ == "__main__":
    main()
