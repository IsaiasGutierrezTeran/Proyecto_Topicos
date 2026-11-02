"""Pruebas rápidas sin Internet ni dataset descargado."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.models.adaptador_nlp import AdaptadorNLP
from src.models.preparar_datos import registro_a_ejemplo


def test_etiquetas_originales_0_a_4():
    assert registro_a_ejemplo({"text": "No me gustó nada", "label": 0})[2] == "negativo"
    assert registro_a_ejemplo({"text": "Fue más o menos", "label": 2})[2] == "neutro"
    assert registro_a_ejemplo({"text": "Muy recomendable", "label": 4})[2] == "positivo"


def test_estrellas_del_corpus_original():
    assert registro_a_ejemplo({"review_body": "Muy buen producto", "stars": 5})[2] == "positivo"


def test_transformacion_basica():
    textos = ["La atención fue muy buena", "No me gustó"]
    assert AdaptadorNLP(metodo="basico").transform(textos) == textos


def test_modelo_entrena_y_predice():
    X = ["excelente maravilloso", "muy bueno", "todo perfecto",
         "terrible horrible", "muy malo", "nada sirve",
         "normal promedio", "regular aceptable", "ni bueno ni malo"]
    y = ["positivo"] * 3 + ["negativo"] * 3 + ["neutro"] * 3
    modelo = Pipeline([("nlp", AdaptadorNLP()),
                       ("tfidf", TfidfVectorizer()),
                       ("clasificador", MultinomialNB())])
    modelo.fit(X, y)
    etiqueta = modelo.predict(["perfecto"])[0]
    assert etiqueta in {"positivo", "negativo", "neutro"}
    assert abs(sum(modelo.predict_proba(["perfecto"])[0]) - 1) < 1e-6
