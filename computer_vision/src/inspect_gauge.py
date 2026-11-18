
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]

IMAGES_DIR = (
    ROOT / "computer_vision/dataset/roboflow_export/valid/images"
)

OUTPUT_DIR = ROOT / "computer_vision/results/gauge_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(IMAGES_DIR.glob("img_0004*"))

if not images:
    raise FileNotFoundError("Imagem img_0004 nao encontrada.")

image = cv2.imread(str(images[0]))

if image is None:
    raise RuntimeError("Nao foi possivel abrir a imagem.")

# Recorte aproximado da região das réguas.
# Ajuste as coordenadas se necessário.
height, width = image.shape[:2]
crop = image[0:min(330, height), 0:min(280, width)]

zoom = cv2.resize(
    crop,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC,
)

output_path = OUTPUT_DIR / "img_0004_zoom.jpg"
cv2.imwrite(str(output_path), zoom)

print(f"Imagem ampliada salva em: {output_path}")
