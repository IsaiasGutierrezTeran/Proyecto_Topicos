"""Prueba de contrato con textos artificiales SOLO de unit testing."""
import tempfile
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.models.adaptador_nlp import AdaptadorNLP


def test_api_joblib_predict_predict_proba_classes():
    X = ["bueno excelente", "bueno maravilloso", "buen producto",
         "malo horrible", "malo pésimo", "pésimo producto",
         "normal aceptable", "regular aceptable", "normal promedio"]
    y = ["positivo"] * 3 + ["negativo"] * 3 + ["neutro"] * 3
    pipeline = Pipeline([
        ("nlp", AdaptadorNLP(metodo="basico")),
        ("tfidf", TfidfVectorizer()),
        ("clasificador", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X,y)
    with tempfile.TemporaryDirectory() as d:
        ruta=Path(d)/"pipeline.joblib"
        joblib.dump(pipeline,ruta)
        cargado=joblib.load(ruta)
        assert cargado.predict(["excelente"]).shape == (1,)
        assert cargado.predict_proba(["excelente"]).shape == (1,3)
        assert len(cargado.classes_) == 3
