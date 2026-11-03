from fastapi.responses import FileResponse
import json
import csv
import math
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.nlp.preprocesamiento import procesar_texto


BASE_DIR = Path(__file__).resolve().parents[1]
MODELO_PATH = BASE_DIR / "artifacts" / "pipeline.joblib"
METRICAS_PATH = BASE_DIR / "artifacts" / "metricas.json"
EXPERIMENTOS_PATH = BASE_DIR / "resultados" / "comparacion_modelos.csv"

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

    try:
        prediccion = modelo.predict([entrada.texto])[0]
        probabilidades = {
            str(clase): float(probabilidad)
            for clase, probabilidad in zip(
                modelo.classes_,
                modelo.predict_proba([entrada.texto])[0]
            )
        }
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

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


@app.get("/experimentos")
def obtener_experimentos():
    """Resultados de CV originales de Kelly, sin reentrenar ni modificar artefactos."""
    campos_texto = ("modelo", "preprocesamiento", "ngramas")
    campos_puntaje = ("f1_macro_cv", "desviacion_cv", "fold_1", "fold_2", "fold_3")
    try:
        with EXPERIMENTOS_PATH.open(encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            requeridos = set(campos_texto + campos_puntaje + ("hiperparametro",))
            if not requeridos.issubset(lector.fieldnames or []):
                raise ValueError("Columnas incompletas")
            filas = []
            for fila in lector:
                salida = {campo: fila[campo].strip() for campo in campos_texto}
                if not all(salida.values()):
                    raise ValueError("Configuración incompleta")
                for campo in campos_puntaje:
                    valor = float(fila[campo])
                    if not math.isfinite(valor) or not 0 <= valor <= 1:
                        raise ValueError("Puntaje inválido")
                    salida[campo] = valor
                parametro = fila["hiperparametro"].strip()
                salida["hiperparametro"] = float(parametro) if parametro else None
                if parametro and (not math.isfinite(salida["hiperparametro"]) or salida["hiperparametro"] <= 0):
                    raise ValueError("Hiperparámetro inválido")
                filas.append(salida)
            if not filas:
                raise ValueError("Sin resultados")
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Comparación experimental pendiente de incorporar") from error
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, csv.Error) as error:
        raise HTTPException(status_code=503, detail="Los resultados experimentales están incompletos o no son válidos.") from error
    return {"fuente": "resultados/comparacion_modelos.csv", "experimentos": filas}


@app.get("/", include_in_schema=False)
def mostrar_frontend():
    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )

