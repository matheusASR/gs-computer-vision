
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "computer_vision" / "results"
    / "training" / "alerta_corrego_yolo11n"
    / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "computer_vision" / "dataset"
    / "roboflow_export" / "data.yaml"
)

RESULTS_DIR = ROOT / "computer_vision" / "results"


def main():
    model = YOLO(str(MODEL_PATH))

    print("\n===== AVALIAÇÃO COM RESOLUÇÃO 640 =====")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(RESULTS_DIR),
        name="validation_640"
    )

    print("\n===== RESULTADOS =====")
    print("mAP50 geral:", metrics.box.map50)
    print("mAP50-95 geral:", metrics.box.map)

    print("\n===== RESULTADO POR CLASSE =====")
    for i, name in enumerate(["staff_gauge", "water_level"]):
        print(f"\n{name}")
        print("mAP50:", metrics.box.maps[i])


if __name__ == "__main__":
    main()
