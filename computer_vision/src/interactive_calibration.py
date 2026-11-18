
from pathlib import Path
import cv2
from ultralytics import YOLO

from estimate_water_level import MODEL_PATH, IMAGES_DIR

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "computer_vision/results/interactive_calibration"

# Utilizaremos a imagem com as marcações visíveis.
IMAGE_PATTERN = "img_0004*"

# Valores de referência da régua central.
REFERENCE_TOP = 85.00
REFERENCE_BOTTOM = 84.00

points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"Ponto {len(points)}: x={x}, y={y}")


def main():
    images = sorted(IMAGES_DIR.glob(IMAGE_PATTERN))

    if not images:
        raise FileNotFoundError("Imagem img_0004 nao encontrada.")

    image_path = images[0]
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("Nao foi possivel abrir a imagem.")

    print("\n===== CALIBRACAO INTERATIVA =====")
    print("Clique na marca de referencia 85.00.")
    print("Depois clique na marca de referencia 84.00.")
    print("Pressione ENTER para confirmar.")
    print("Pressione R para reiniciar ou ESC para sair.")

    window_name = "Calibracao da Regua"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = image.copy()

        for index, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"P{index + 1}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            return

        if key == ord("r"):
            points.clear()

        if key in (13, 10) and len(points) == 2:
            break

    cv2.destroyAllWindows()

    (x_top, y_top), (x_bottom, y_bottom) = points

    if y_top == y_bottom:
        raise ValueError("As referencias nao podem ter o mesmo Y.")

    model = YOLO(str(MODEL_PATH))

    result = model.predict(
        source=image,
        imgsz=416,
        conf=0.15,
        device="cpu",
        verbose=False,
    )[0]

    # Seleciona a detecção de água mais próxima
    # horizontalmente do ponto médio das referências.
    reference_x = (x_top + x_bottom) / 2

    candidates = []

    for detection in result.boxes:
        if int(detection.cls[0]) != 1:
            continue

        box = detection.xyxy[0].cpu().numpy()
        confidence = float(detection.conf[0])

        water_x = (box[0] + box[2]) / 2
        water_y = (box[1] + box[3]) / 2

        candidates.append((
            abs(water_x - reference_x),
            water_y,
            confidence,
            box,
        ))

    if not candidates:
        print("Nenhuma linha de agua detectada.")
        return

    candidates.sort(key=lambda item: item[0])

    _, water_y, confidence, water_box = candidates[0]

    minimum_y = min(y_top, y_bottom)
    maximum_y = max(y_top, y_bottom)

    if not minimum_y <= water_y <= maximum_y:
        print(
            "AVISO: linha da agua fora do intervalo "
            "de calibracao. Resultado extrapolado."
        )

    # Interpolação linear entre as duas referências.
    estimated_level = REFERENCE_TOP + (
        (water_y - y_top)
        * (REFERENCE_BOTTOM - REFERENCE_TOP)
        / (y_bottom - y_top)
    )

    output = image.copy()

    cv2.circle(output, (x_top, y_top), 5, (0, 255, 255), -1)
    cv2.circle(output, (x_bottom, y_bottom), 5, (0, 255, 255), -1)

    x1, _, x2, _ = map(int, water_box)

    cv2.line(
        output,
        (x1, int(water_y)),
        (x2, int(water_y)),
        (255, 0, 0),
        3,
    )

    cv2.putText(
        output,
        f"Leitura estimada: {estimated_level:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), output)

    print("\n===== RESULTADO =====")
    print(f"Referencia superior: y={y_top}, valor={REFERENCE_TOP}")
    print(f"Referencia inferior: y={y_bottom}, valor={REFERENCE_BOTTOM}")
    print(f"Linha de agua: y={water_y:.1f}")
    print(f"Confianca da deteccao: {confidence:.2f}")
    print(f"Leitura estimada: {estimated_level:.2f}")
    print(f"Imagem salva em: {output_path}")


if __name__ == "__main__":
    main()
