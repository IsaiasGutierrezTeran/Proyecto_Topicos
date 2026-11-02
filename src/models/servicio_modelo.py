"""Servicio de predicción del modelo OpinIA."""

from functools import lru_cache
from pathlib import Path

import joblib


RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELO = RAIZ / "artifacts" / "pipeline.joblib"


@lru_cache(maxsize=1)
def cargar_pipeline():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    # Cargar exclusivamente modelos creados por nuestro equipo.
    return joblib.load(RUTA_MODELO)


def predecir_sentimiento(texto: str):
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("Debes ingresar una reseña válida.")

    modelo = cargar_pipeline()

    sentimiento = str(modelo.predict([texto])[0])

    probabilidades = modelo.predict_proba([texto])[0]

    clases = [
        str(clase)
        for clase in modelo.classes_
    ]

    return {
        "sentimiento": sentimiento,
        "probabilidades": {
            clase: float(probabilidad)
            for clase, probabilidad in zip(
                clases, probabilidades
            )
        }
    }