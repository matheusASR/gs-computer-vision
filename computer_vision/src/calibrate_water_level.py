
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "computer_vision/results/calibration"


def calculate_height(water_y, y_top, cm_top, y_bottom, cm_bottom):
    if y_top == y_bottom:
        raise ValueError("As referencias precisam ter alturas diferentes.")

    return cm_top + (
        (water_y - y_top)
        * (cm_bottom - cm_top)
        / (y_bottom - y_top)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--y-top", type=float, required=True)
    parser.add_argument("--cm-top", type=float, required=True)
    parser.add_argument("--y-bottom", type=float, required=True)
    parser.add_argument("--cm-bottom", type=float, required=True)

    args = parser.parse_args()

    image_path = IMAGES_DIR / args.image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    model = YOLO(str(MODEL_PATH))

    result = model.predict(
        source=image,
        imgsz=416,
        conf=0.15,
        device="cpu",
        verbose=False,
    )[0]

    water_detections = []

    for detection in result.boxes:
        if int(detection.cls[0]) != 1:
            continue

        box = detection.xyxy[0].cpu().numpy()
        confidence = float(detection.conf[0])

        water_detections.append((box, confidence))

    if not water_detections:
        print("Nenhuma linha de agua detectada.")
        return

    # Protótipo: utiliza a detecção de água
    # com maior confiança na imagem.
    water_box, confidence = max(
        water_detections,
        key=lambda item: item[1]
    )

    x1, y1, x2, y2 = map(int, water_box)
    water_y = (y1 + y2) / 2

    estimated_height = calculate_height(
        water_y,
        args.y_top,
        args.cm_top,
        args.y_bottom,
        args.cm_bottom,
    )

    output = image.copy()

    cv2.line(
        output,
        (x1, int(water_y)),
        (x2, int(water_y)),
        (255, 0, 0),
        3,
    )

    cv2.putText(
        output,
        f"Nivel estimado: {estimated_height:.1f} cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), output)

    print(f"Confianca da agua: {confidence:.2f}")
    print(f"Linha detectada em y={water_y:.1f}")
    print(f"Nivel estimado: {estimated_height:.1f} cm")
    print(f"Imagem salva em: {output_path}")


if __name__ == "__main__":
    main()
