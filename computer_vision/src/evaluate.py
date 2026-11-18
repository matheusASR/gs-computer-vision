
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "computer_vision" / "results" / "training"
    / "alerta_corrego_yolo11n" / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "computer_vision" / "dataset"
    / "roboflow_export" / "data.yaml"
)

TEST_IMAGES = (
    ROOT / "computer_vision" / "dataset"
    / "roboflow_export" / "test" / "images"
)

RESULTS_DIR = ROOT / "computer_vision" / "results"

def main():
    model = YOLO(str(MODEL_PATH))

    print("\n===== AVALIAÇÃO NO TESTE =====")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=str(RESULTS_DIR),
        name="test_evaluation"
    )

    print("\n===== MÉTRICAS =====")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\n===== GERANDO DETECÇÕES =====")

    model.predict(
        source=str(TEST_IMAGES),
        imgsz=416,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(RESULTS_DIR),
        name="test_predictions"
    )

    print("\nAvaliação concluída!")

if __name__ == "__main__":
    main()
