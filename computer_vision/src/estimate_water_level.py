
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "computer_vision" / "results"
    / "training" / "alerta_corrego_yolo11n"
    / "weights" / "best.pt"
)

IMAGES_DIR = (
    ROOT / "computer_vision" / "dataset"
    / "roboflow_export" / "valid" / "images"
)

OUTPUT_DIR = (
    ROOT / "computer_vision" / "results"
    / "water_level_estimation"
)


def estimate_water_line(image, box):
    """
    Estima uma possível linha da água dentro da régua,
    usando variações verticais de intensidade.
    """
    x1, y1, x2, y2 = map(int, box)

    height, width = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = image[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < 10:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Intensidade média de cada linha horizontal
    row_mean = np.mean(gray.astype(np.float32), axis=1)

    # Diferença entre linhas consecutivas
    differences = np.abs(np.diff(row_mean))

    # Busca a transição na metade inferior da régua
    start = int(len(differences) * 0.5)

    if start >= len(differences):
        return None

    relative_y = np.argmax(differences[start:])
    water_y = y1 + start + int(relative_y)

    return water_y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    images = list(IMAGES_DIR.glob("*.jpg"))[:10]

    if not images:
        print("Nenhuma imagem encontrada.")
        return

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        results = model.predict(
            source=image,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]
        output = image.copy()

        for box in result.boxes:
            class_id = int(box.cls[0])

            # Processar apenas a classe staff_gauge
            if class_id != 0:
                continue

            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)

            confidence = float(box.conf[0])

            # Desenhar a régua detectada
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"staff_gauge {confidence:.2f}",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            water_y = estimate_water_line(image, coords)

            if water_y is not None:
                cv2.line(
                    output,
                    (x1, water_y),
                    (x2, water_y),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    f"Water line: y={water_y}",
                    (x1, min(water_y + 20, output.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(str(output_path), output)

        print(f"Processado: {image_path.name}")

    print("\nEstimativas salvas em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
