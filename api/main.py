from fastapi.responses import FileResponse
import json
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.nlp.preprocesamiento import procesar_texto


BASE_DIR = Path(__file__).resolve().parents[1]
MODELO_PATH = BASE_DIR / "artifacts" / "pipeline.joblib"
METRICAS_PATH = BASE_DIR / "artifacts" / "metricas.json"

app = FastAPI(
    title="Sistema de Análisis de Sentimientos",
    description="API de clasificación de reseñas mediante NLP",
    version="1.0.0",
)


class EntradaTexto(BaseModel):
    texto: str = Field(min_length=3, max_length=5000)


@app.get("/health")
def verificar_estado():
    return {
        "estado": "activo",
        "modelo_disponible": MODELO_PATH.is_file()
    }


@app.post("/preprocesar")
def preprocesar(entrada: EntradaTexto):
    try:
        return procesar_texto(entrada.texto)
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error


@lru_cache(maxsize=1)
def cargar_modelo():
    if not MODELO_PATH.is_file():
        raise FileNotFoundError("Modelo no disponible")
    return joblib.load(MODELO_PATH)


@app.post("/predecir")
def predecir(entrada: EntradaTexto):
    if not entrada.texto.strip():
        raise HTTPException(
            status_code=422,
            detail="El texto no puede estar vacío"
        )

    try:
        modelo = cargar_modelo()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está entrenado"
        )

    prediccion = modelo.predict([entrada.texto])[0]

    probabilidades = {
        str(clase): float(probabilidad)
        for clase, probabilidad in zip(
            modelo.classes_,
            modelo.predict_proba([entrada.texto])[0]
        )
    }

    return {
        "sentimiento": str(prediccion),
        "probabilidades": probabilidades
    }



@app.get("/metricas")
def obtener_metricas():
    if not METRICAS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Las métricas todavía no están disponibles"
        )

    with open(METRICAS_PATH, "r", encoding="utf-8") as archivo:
        metricas = json.load(archivo)

    if metricas is None:
        raise HTTPException(
            status_code=503,
            detail="El archivo de métricas está vacío"
        )

    return metricas


@app.get("/", include_in_schema=False)
def mostrar_frontend():
    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )

