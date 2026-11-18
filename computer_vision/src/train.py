
from pathlib import Path
from ultralytics import YOLO

# Localização do projeto
ROOT = Path(__file__).resolve().parents[2]

# Dataset
DATA_YAML = (
    ROOT
    / "computer_vision"
    / "dataset"
    / "roboflow_export"
    / "data.yaml"
)

# Diretório dos resultados
OUTPUT_DIR = ROOT / "computer_vision" / "results" / "training"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Arquivo data.yaml não encontrado: {DATA_YAML}"
        )

    print("Carregando modelo YOLO11n...")

    # Carrega pesos pré-treinados
    model = YOLO("yolo11n.pt")

    print("Iniciando treinamento do Alerta Córrego...")

    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUTPUT_DIR),
        name="alerta_corrego_yolo11n",
        patience=10,
        cache=False,
        plots=True,
        save=True
    )

    print("\nTreinamento concluído!")
    print(f"Resultados disponíveis em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
