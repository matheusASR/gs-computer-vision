
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

# Caminhos
PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_DIR / "computer_vision" / "dataset" / "images" / "raw"
OUTPUT_DIR = PROJECT_DIR / "computer_vision" / "results"
OUTPUT_FILE = OUTPUT_DIR / "dataset_preview.jpg"

# Configurações do painel
COLUMNS = 5
THUMB_WIDTH = 240
THUMB_HEIGHT = 180
LABEL_HEIGHT = 30
PADDING = 12

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Localiza imagens
extensions = {".jpg", ".jpeg", ".png"}
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

if not image_paths:
    raise SystemExit(f"Nenhuma imagem encontrada em: {IMAGE_DIR}")

rows = math.ceil(len(image_paths) / COLUMNS)

cell_width = THUMB_WIDTH
cell_height = THUMB_HEIGHT + LABEL_HEIGHT

sheet_width = COLUMNS * cell_width + (COLUMNS + 1) * PADDING
sheet_height = rows * cell_height + (rows + 1) * PADDING

sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
draw = ImageDraw.Draw(sheet)

for index, image_path in enumerate(image_paths):
    try:
        image = Image.open(image_path).convert("RGB")
        thumbnail = ImageOps.fit(
            image,
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        row = index // COLUMNS
        column = index % COLUMNS

        x = PADDING + column * (cell_width + PADDING)
        y = PADDING + row * (cell_height + PADDING)

        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x, y + THUMB_HEIGHT + 5),
            image_path.name,
            fill="black"
        )

    except Exception as error:
        print(f"Erro em {image_path.name}: {error}")

sheet.save(OUTPUT_FILE, quality=90)

print(f"Imagens encontradas: {len(image_paths)}")
print(f"Painel salvo em: {OUTPUT_FILE}")
