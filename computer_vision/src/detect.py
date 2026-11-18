"""Executa inferência YOLO em imagem, vídeo ou webcam."""
import argparse
from ultralytics import YOLO



def parse_args():
    parser = argparse.ArgumentParser(description="Detecção visual - Alerta Córrego")
    parser.add_argument("--source", default="0", help="Arquivo de imagem/vídeo ou índice da webcam (0)")
    parser.add_argument("--weights", required=True, help="Caminho do arquivo de pesos .pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima da detecção")
    parser.add_argument("--save", action="store_true", help="Salvar imagens/vídeos com detecções")
    return parser.parse_args()



def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    model.predict(source=source, conf=args.conf, show=True, save=args.save)



if __name__ == "__main__":
    main()
