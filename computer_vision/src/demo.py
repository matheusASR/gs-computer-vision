
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "computer_vision/results/demo"

CONFIDENCE = 0.15
MIN_GAUGE_CONFIDENCE = 0.30


def center_x(box):
    return (box[0] + box[2]) / 2


def center_y(box):
    return (box[1] + box[3]) / 2


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
            margin = 0.10 * gauge_height

            if distance > gauge_width:
                continue

            if not gy1 - margin <= wy <= gy2 + margin:
                continue

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


def process_image(model, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Erro ao abrir: {image_path}")
        return

    result = model.predict(
        source=image,
        imgsz=416,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False,
    )[0]

    gauges = []
    waters = []

    for detection in result.boxes:
        box = detection.xyxy[0].cpu().numpy()
        confidence = float(detection.conf[0])
        class_id = int(detection.cls[0])

        if class_id == 0 and confidence >= MIN_GAUGE_CONFIDENCE:
            gauges.append((box, confidence))

        elif class_id == 1:
            waters.append((box, confidence))

    associations = associate_water(gauges, waters)

    output = image.copy()

    # Desenha as réguas.
    for box, confidence in gauges:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    # Desenha apenas linhas de água associadas.
    for gi, wi in associations:
        gauge, gauge_conf = gauges[gi]
        water, water_conf = waters[wi]

        gx1, gy1, gx2, gy2 = map(int, gauge)
        water_y = int(center_y(water))

        cv2.line(
            output,
            (gx1, water_y),
            (gx2, water_y),
            (255, 0, 0),
            3,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), output)

    print(
        f"{image_path.name}: "
        f"{len(gauges)} regua(s), "
        f"{len(associations)} linha(s) de agua associada(s)"
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Demonstracao do Alerta Corrego"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Nome da imagem. Se omitido, processa todas."
    )

    args = parser.parse_args()

    model = YOLO(str(MODEL_PATH))

    if args.image:
        image_paths = [IMAGES_DIR / args.image]
    else:
        image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    if not image_paths:
        print("Nenhuma imagem encontrada.")
        return

    print("\n===== ALERTA CORREGO =====")
    print(f"Modelo: {MODEL_PATH.name}")
    print(f"Confianca minima: {CONFIDENCE}")
    print(f"Imagens selecionadas: {len(image_paths)}\n")

    for image_path in image_paths:
        if not image_path.is_file():
            print(f"Arquivo nao encontrado: {image_path}")
            continue

        process_image(model, image_path)

    print(f"\nResultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
