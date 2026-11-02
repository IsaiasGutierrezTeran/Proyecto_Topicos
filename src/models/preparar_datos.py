"""Descarga una muestra reproducible de reseñas reales en español.

Uso (desde la raíz del repositorio):
    python -m src.models.preparar_datos

El dataset original permanece en Hugging Face. No subir los CSV de reseñas a GitHub
sin verificar los términos de uso. Los CSV se recrean ejecutando este script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
BASE_URL = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/es"
ROOT = Path(__file__).resolve().parents[2]
DESTINO = ROOT / "data" / "processed"


def registro_a_ejemplo(registro: dict) -> tuple[str, int, str] | None:
    """Retorna (texto, estrellas 1..5, sentimiento), validando las columnas."""
    texto = registro.get("text") or registro.get("review_body")
    if not isinstance(texto, str) or len(texto.strip()) < 8:
        return None
    if "label" in registro:  # En mteb: label de 0 a 4
        estrellas = int(registro["label"]) + 1
    elif "stars" in registro:  # Formato del corpus original: de 1 a 5
        estrellas = int(registro["stars"])
    else:
        raise ValueError("El archivo no contiene la columna 'label' ni 'stars'.")
    if not 1 <= estrellas <= 5:
        raise ValueError(f"Puntuación inesperada: {estrellas}")
    sentimiento = "negativo" if estrellas <= 2 else "neutro" if estrellas == 3 else "positivo"
    return texto.strip(), estrellas, sentimiento


def seleccionar(split: str, por_clase: int) -> tuple[list[dict], int]:
    """Muestreo reservoir por estrella sobre TODO el split, con semilla fija."""
    if por_clase % 2:
        raise ValueError("--por-clase debe ser par para distribuir estrellas 1/2 y 4/5.")
    limite = {1: por_clase // 2, 2: por_clase // 2, 3: por_clase,
              4: por_clase // 2, 5: por_clase // 2}
    seen = Counter()
    depositos: dict[int, list[dict]] = {i: [] for i in range(1, 6)}
    rng = random.Random(SEED + (0 if split == "train" else 1))
    url = f"{BASE_URL}/{split}.jsonl"
    peticion = urllib.request.Request(url, headers={"User-Agent": "ProyectoTopicosNLP/1.0 (academico)"})
    print(f"Leyendo reseñas de {split} desde {url}")
    try:
        with urllib.request.urlopen(peticion, timeout=120) as respuesta:
            for linea in respuesta:
                if not linea.strip():
                    continue
                ejemplo = registro_a_ejemplo(json.loads(linea.decode("utf-8-sig")))
                if ejemplo is None:
                    continue
                texto, estrellas, sentimiento = ejemplo
                seen[estrellas] += 1
                item = {"texto": texto, "sentimiento": sentimiento, "estrellas": estrellas}
                grupo = depositos[estrellas]
                if len(grupo) < limite[estrellas]:
                    grupo.append(item)
                else:
                    pos = rng.randrange(seen[estrellas])
                    if pos < limite[estrellas]:
                        grupo[pos] = item
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo descargar {url}. Revisa tu conexión. También puedes "
            "descargar ese JSONL desde el navegador y usar otro origen documentado."
        ) from exc
    faltantes = {s: limite[s] - len(depositos[s]) for s in range(1, 6)
                 if len(depositos[s]) < limite[s]}
    if faltantes:
        raise RuntimeError(f"Faltan reseñas suficientes por estrella: {faltantes}")
    muestras = [ej for grupo in depositos.values() for ej in grupo]
    rng.shuffle(muestras)
    return muestras, sum(seen.values())


def guardar_csv(ruta: Path, muestras: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["texto", "sentimiento", "estrellas"])
        escritor.writeheader()
        escritor.writerows(muestras)


def main() -> None:
    p = argparse.ArgumentParser(description="Preparar reseñas reales en español")
    p.add_argument("--train-por-clase", type=int, default=200)
    p.add_argument("--test-por-clase", type=int, default=60)
    args = p.parse_args()
    train, total_train = seleccionar("train", args.train_por_clase)
    test, total_test = seleccionar("test", args.test_por_clase)
    # Eliminar duplicados textuales entre conjuntos y dentro de cada conjunto.
    vistos = set()
    def sin_duplicados(muestras):
        salida = []
        for item in muestras:
            llave = item["texto"].casefold().strip()
            if llave not in vistos:
                vistos.add(llave)
                salida.append(item)
        return salida
    train = sin_duplicados(train)
    test = sin_duplicados(test)
    guardar_csv(DESTINO / "train.csv", train)
    guardar_csv(DESTINO / "test.csv", test)
    metadata = {
        "fuente": "mteb/amazon_reviews_multi", "idioma": "es",
        "url": "https://huggingface.co/datasets/mteb/amazon_reviews_multi",
        "url_train": f"{BASE_URL}/train.jsonl", "url_test": f"{BASE_URL}/test.jsonl",
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(),
        "semilla": SEED, "split_test_independiente": True,
        "total_origen_train": total_train, "total_origen_test": total_test,
        "tamano_train": len(train), "tamano_test": len(test),
        "distribucion_train": dict(Counter(x["sentimiento"] for x in train)),
        "distribucion_test": dict(Counter(x["sentimiento"] for x in test)),
        "criterio": "1-2 estrellas=negativo; 3=neutro; 4-5=positivo",
        "licencia": "NO VERIFICADA en la ficha del espejo mteb al 2026-11-02",
        "nota_licencia": "Revisar y documentar las condiciones del corpus original antes de redistribuir reseñas."
    }
    ruta_meta = ROOT / "docs" / "metadata_dataset.json"
    ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    ruta_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dataset preparado:")
    print(f"  Entrenamiento: {len(train)} reseñas, {metadata['distribucion_train']}")
    print(f"  Prueba: {len(test)} reseñas, {metadata['distribucion_test']}")
    print(f"  Archivos: {DESTINO}/train.csv y test.csv")


if __name__ == "__main__":
    main()
