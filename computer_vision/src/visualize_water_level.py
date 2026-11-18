
from pathlib import Path

import cv2
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "computer_vision/results/yolo_water_final"

CONFIDENCE = 0.15
MIN_GAUGE_CONFIDENCE = 0.30


def center_x(box):
    return (box[0] + box[2]) / 2


def center_y(box):
    return (box[1] + box[3]) / 2


def horizontal_overlap(box_a, box_b):
    left = max(box_a[0], box_b[0])
    right = min(box_a[2], box_b[2])
    return max(0, right - left)


def associate_water(gauges, waters):
    candidates = []

    for gi, (gauge, _) in enumerate(gauges):
        gx1, gy1, gx2, gy2 = gauge

        gauge_width = max(1, gx2 - gx1)
        gauge_height = max(1, gy2 - gy1)

        for wi, (water, confidence) in enumerate(waters):
            wx = center_x(water)
            wy = center_y(water)

            distance = abs(wx - center_x(gauge))

            # A água deve estar alinhada horizontalmente
            # com a régua, considerando pequena tolerância.
            if distance > gauge_width:
                continue

            # A linha deve estar dentro ou próxima da régua.
            margin = 0.10 * gauge_height

            if not gy1 - margin <= wy <= gy2 + margin:
                continue

            # Prefere proximidade horizontal e confiança alta.
            score = distance / gauge_width - 0.2 * confidence
            candidates.append((score, gi, wi))

    candidates.sort()

    associations = []
    used_gauges = set()
    used_waters = set()

    for _, gi, wi in candidates:
        if gi in used_gauges or wi in used_waters:
            continue

        associations.append((gi, wi))
        used_gauges.add(gi)
        used_waters.add(wi)

    return associations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    images = sorted(IMAGES_DIR.glob("*.jpg"))

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        output = image.copy()

        result = model.predict(
            source=image,
            imgsz=416,
            conf=CONFIDENCE,
            iou=0.45,
            device="cpu",
            verbose=False,
        )[0]

        gauges = []
        waters = []

        for detection in result.boxes:
            box = detection.xyxy[0].cpu().numpy()
            confidence = float(detection.conf[0])
            class_id = int(detection.cls[0])

            if class_id == 0:
                if confidence >= MIN_GAUGE_CONFIDENCE:
                    gauges.append((box, confidence))

            elif class_id == 1:
                waters.append((box, confidence))

        associations = associate_water(gauges, waters)

        # Desenha todas as réguas consideradas válidas.
        for box, confidence in gauges:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        # Desenha somente as águas associadas.
        for gi, wi in associations:
            gauge, gauge_conf = gauges[gi]
            water, water_conf = waters[wi]

            gx1, gy1, gx2, gy2 = map(int, gauge)
            water_y = int(center_y(water))

            # Linha horizontal na largura da régua.
            cv2.line(
                output,
                (gx1, water_y),
                (gx2, water_y),
                (255, 0, 0),
                2,
            )

            label = f"Agua {water_conf:.2f}"

            cv2.putText(
                output,
                label,
                (gx1, max(15, gy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), output)

        print(
            f"{image_path.name}: "
            f"{len(gauges)} reguas, "
            f"{len(associations)} linhas associadas"
        )

    print("\nResultados salvos em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
