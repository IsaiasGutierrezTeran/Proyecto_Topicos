"""Adaptador entre el NLP de Isaías y el Pipeline de scikit-learn de Kelly.

No modifica src/nlp/preprocesamiento.py. Ese módulo ofrece procesar_texto(texto)
y debe devolver ambos resultados (stemming y lematización) en un diccionario.
"""
from functools import lru_cache
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin


def _clave_normalizada(clave):
    sin_tildes = unicodedata.normalize("NFKD", str(clave))
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.casefold().replace("-", "_").replace(" ", "_")


@lru_cache(maxsize=3000)
def _nlp_isaias(texto):
    # Importación diferida para que el modo 'basico' no requiera spaCy.
    from src.nlp.preprocesamiento import procesar_texto
    return procesar_texto(texto)


class AdaptadorNLP(BaseEstimator, TransformerMixin):
    def __init__(self, metodo="basico"):
        self.metodo = metodo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        textos = [str(texto) for texto in X]
        if self.metodo == "basico":
            return textos
        if self.metodo not in {"stemming", "lematizacion"}:
            raise ValueError(f"Método desconocido: {self.metodo}")

        alias = {
            "stemming": ("stemming", "stem", "stems", "texto_stemming", "tokens_stemming", "raices"),
            "lematizacion": ("lematizacion", "lematización", "lemas", "lemma", "lemmas", "lematizado", "texto_lematizado", "tokens_lematizados"),
        }[self.metodo]
        alias = {_clave_normalizada(v) for v in alias}
        salidas = []
        for texto in textos:
            resultado = _nlp_isaias(texto)
            if not isinstance(resultado, dict):
                raise TypeError(
                    "procesar_texto() debe devolver un diccionario con resultados "
                    "separados para stemming y lematización. "
                    f"Devolvió {type(resultado).__name__}."
                )
            etapas = resultado
            # Soporte para un diccionario de etapas anidado.
            for clave in ("etapas", "procesamiento", "resultados"):
                if isinstance(etapas.get(clave), dict):
                    etapas = etapas[clave]
                    break
            clave = next((k for k in etapas if _clave_normalizada(k) in alias), None)
            if clave is None:
                raise KeyError(
                    f"No se encontró el resultado de '{self.metodo}' en "
                    f"procesar_texto(). Claves existentes: {list(etapas)}. "
                    "Hay que ajustar el alias en src/models/adaptador_nlp.py "
                    "a la salida real de Isaías."
                )
            valor = etapas[clave]
            if isinstance(valor, (list, tuple)):
                valor = " ".join(str(v) for v in valor)
            if not isinstance(valor, str):
                raise TypeError(
                    f"El resultado '{clave}' debe ser texto o lista de tokens, "
                    f"no {type(valor).__name__}."
                )
            salidas.append(valor)
        return salidas
