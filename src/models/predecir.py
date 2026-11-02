"""Probador de terminal del modelo entrenado por Kelly."""
import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texto", help="Reseña para clasificar; ponla entre comillas")
    args = parser.parse_args()
    modelo = joblib.load(ROOT / "artifacts" / "pipeline.joblib")
    etiqueta = str(modelo.predict([args.texto])[0])
    probabilidades = modelo.predict_proba([args.texto])[0]
    print("Sentimiento:", etiqueta)
    for clase, probabilidad in zip(modelo.classes_, probabilidades):
        print(f"  {clase}: {probabilidad:.2%}")


if __name__ == "__main__":
    main()
