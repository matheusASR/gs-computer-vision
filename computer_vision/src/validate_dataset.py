
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "computer_vision" / "dataset" / "roboflow_export"

splits = ["train", "valid", "test"]
class_counts = Counter()
errors = []
empty_labels = 0

for split in splits:
    labels_dir = DATASET / split / "labels"
    files = list(labels_dir.glob("*.txt"))

    print(f"\nValidando {split}: {len(files)} arquivos")

    for file in files:
        lines = file.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(f"{file.name}, linha {line_number}: formato inválido")
                continue

            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(f"{file.name}, linha {line_number}: valores inválidos")
                continue

            if class_id not in [0, 1]:
                errors.append(f"{file.name}: classe desconhecida {class_id}")
                continue

            if not all(0 <= value <= 1 for value in [x, y, width, height]):
                errors.append(f"{file.name}: coordenadas fora do intervalo")
                continue

            if width <= 0 or height <= 0:
                errors.append(f"{file.name}: caixa com dimensão inválida")
                continue

            class_counts[class_id] += 1

print("\n========== RESULTADO ==========")
print("staff_gauge:", class_counts[0])
print("water_level:", class_counts[1])
print("Labels vazios:", empty_labels)
print("Erros encontrados:", len(errors))

for error in errors[:20]:
    print("ERRO:", error)

if not errors:
    print("\nAnotações válidas!")
