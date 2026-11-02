"""Genera un pequeño EDA reproducible sin notebooks ni Jupyter.
Ejecuta: python -m src.models.eda
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "resultados"


def main():
    train = ROOT / "data" / "processed" / "train.csv"
    test = ROOT / "data" / "processed" / "test.csv"
    if not train.exists() or not test.exists():
        raise FileNotFoundError("Faltan los CSV: ejecuta primero python -m src.models.preparar_datos")
    df_train = pd.read_csv(train, encoding="utf-8-sig")
    df_test = pd.read_csv(test, encoding="utf-8-sig")
    df = pd.concat([df_train.assign(conjunto="train"), df_test.assign(conjunto="test")], ignore_index=True)
    if "texto" not in df.columns or "sentimiento" not in df.columns:
        raise ValueError("Se requieren las columnas texto y sentimiento")
    df["longitud_caracteres"] = df["texto"].fillna("").astype(str).str.len()
    df["longitud_palabras"] = df["texto"].fillna("").astype(str).str.split().str.len()
    tokens = Counter(
        w for texto in df_train["texto"].fillna("").astype(str)
        for w in re.findall(r"\b[a-záéíóúüñ]{3,}\b", texto.lower())
    )
    resumen = {
        "registros_total": int(len(df)),
        "train": int(len(df_train)), "test": int(len(df_test)),
        "faltantes": {k:int(v) for k,v in df.isna().sum().items()},
        "distribucion_clases": {k:int(v) for k,v in df["sentimiento"].value_counts().items()},
        "distribucion_por_conjunto": {s:{k:int(v) for k,v in g["sentimiento"].value_counts().items()} for s,g in df.groupby("conjunto")},
        "media_palabras": round(float(df["longitud_palabras"].mean()),2),
        "mediana_palabras": round(float(df["longitud_palabras"].median()),2),
        "top_20_palabras_sin_filtrar": [{"palabra":w, "frecuencia":int(n)} for w,n in tokens.most_common(20)],
        "nota": "Top palabras sin stop words, solo frecuencia bruta; consultar NLP de Isaías para el filtrado."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / "eda_resumen.json").write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6,4))
    df["sentimiento"].value_counts().reindex(["negativo","neutro","positivo"],fill_value=0).plot.bar(ax=ax)
    ax.set_title("Distribución de clases"); ax.set_ylabel("Número de reseñas")
    fig.tight_layout(); fig.savefig(OUT/"eda_clases.png",dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4))
    df["longitud_palabras"].plot.hist(ax=ax,bins=25)
    ax.set_title("Longitud de las reseñas"); ax.set_xlabel("Palabras"); ax.set_ylabel("Frecuencia")
    fig.tight_layout(); fig.savefig(OUT/"eda_longitudes.png",dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    frecuentes = tokens.most_common(12)
    ax.barh([w for w,n in frecuentes][::-1],[n for w,n in frecuentes][::-1])
    ax.set_title("Palabras más frecuentes (sin filtrar)"); ax.set_xlabel("Frecuencia")
    fig.tight_layout(); fig.savefig(OUT/"eda_palabras.png",dpi=130); plt.close(fig)
    print("EDA real guardado en resultados/eda_resumen.json y tres gráficos PNG")
    print("Registros:", resumen["registros_total"], "| Clases:",resumen["distribucion_clases"])

if __name__ == "__main__":
    main()
