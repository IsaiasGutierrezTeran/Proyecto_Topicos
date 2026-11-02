
import re
import unicodedata
from functools import lru_cache

import spacy
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.es.stop_words import STOP_WORDS


# Conservamos las negaciones porque pueden cambiar
# el sentimiento de una reseña.
NEGACIONES = {"no", "ni", "nunca", "sin", "tampoco"}

STOPWORDS = set(STOP_WORDS) - NEGACIONES

stemmer = SnowballStemmer("spanish")


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga una única vez el modelo lingüístico español."""
    return spacy.load(
        "es_core_news_sm",
        disable=["parser", "ner"]
    )


def normalizar_texto(texto):
    """Normalización de una reseña en español."""

    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena.")

    texto = unicodedata.normalize("NFC", texto)
    texto = texto.lower()

    # Eliminar enlaces.
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)

    # Eliminar números.
    texto = re.sub(r"\d+", " ", texto)

    # Eliminar emojis y símbolos Unicode.
    texto = "".join(
        caracter for caracter in texto
        if not unicodedata.category(caracter).startswith("S")
    )

    # Eliminar puntuación conservando letras españolas.
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = texto.replace("_", " ")

    # Eliminar espacios duplicados.
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def procesar_texto(texto):
    """Ejecuta y muestra cada etapa del procesamiento NLP."""

    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena.")

    if not texto.strip():
        raise ValueError("El texto no puede estar vacío.")

    normalizado = normalizar_texto(texto)

    if not normalizado:
        raise ValueError("El texto no contiene palabras válidas.")

    modelo = cargar_modelo()
    documento = modelo(normalizado)

    # Tokenización.
    tokens = [
        token.text
        for token in documento
        if token.is_alpha
    ]

    # Eliminación de stop words.
    filtrados = [
        token
        for token in documento
        if token.is_alpha
        and token.text.lower() not in STOPWORDS
    ]

    # Stemming: obtener raíces aproximadas.
    raices = [
        stemmer.stem(token.text)
        for token in filtrados
    ]

    # Lematización: obtener formas base.
    lemas = [
        token.lemma_.lower()
        for token in filtrados
    ]

    return {
        "original": texto,
        "normalizado": normalizado,
        "tokens": tokens,
        "sin_stopwords": [t.text for t in filtrados],
        "stemming": raices,
        "lematizacion": lemas
    }


if __name__ == "__main__":

    ejemplo = (
        "¡NO me gustaron los productos! "
        "La atención fue pésima 😡. "
        "Visita https://ejemplo.com 123"
    )

    resultado = procesar_texto(ejemplo)

    for etapa, valor in resultado.items():
        print(f"\n{etapa.upper()}:")
        print(valor)
