
from pathlib import Path

import cv2
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR
from evaluate_water_line import load_annotations

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT / "computer_vision"
    / "results"
    / "water_comparison"
)

LABELS_DIR = IMAGES_DIR.parent / "labels"

CONFIDENCE = 0.15


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        output = image.copy()

        annotations = load_annotations(
            label_path, width, height
        )

        # Ground truth: vermelho
        for annotation in annotations:
            if annotation["class_id"] != 1:
                continue

            x1, y1, x2, y2 = map(
                int, annotation["box"]
            )

            water_y = int(annotation["yc"])

            cv2.line(
                output,
                (x1, water_y),
                (x2, water_y),
                (0, 0, 255),
                3,
            )

        result = model.predict(
            source=image,
            imgsz=416,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        for detection in result.boxes:
            box = detection.xyxy[0].cpu().numpy()
            confidence = float(detection.conf[0])
            class_id = int(detection.cls[0])

            x1, y1, x2, y2 = map(int, box)

            if class_id == 0:
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            elif class_id == 1:
                water_y = int((y1 + y2) / 2)

                cv2.line(
                    output,
                    (x1, water_y),
                    (x2, water_y),
                    (255, 0, 0),
                    2,
                )

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), output)

        print(f"Processado: {image_path.name}")

    print("\nResultados salvos em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
