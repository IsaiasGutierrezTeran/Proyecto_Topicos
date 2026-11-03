"""Contratos reales de NLP, muestreo, CV e integracion con modelos guardados."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from src.models.adaptador_nlp import AdaptadorNLP
from src.models.preparar_datos import seleccionar
from src.models.entrenar import experimentos
from src.nlp.preprocesamiento import procesar_texto

ROOT = Path(__file__).resolve().parents[1]

def test_adaptador_usa_exactamente_nlp_compartido():
    textos = ["No me gustaron los productos.", "Funciona pero no cumple lo prometido."]
    for metodo in ["stemming", "lematizacion"]:
        esperado = [" ".join(procesar_texto(t)[metodo]) for t in textos]
        assert AdaptadorNLP(metodo).transform(textos) == esperado
        assert "no" in esperado[0].split()

def test_reservoir_reproducible_desde_crudos(tmp_path):
    revision = "fixture"
    carpeta = tmp_path / revision
    carpeta.mkdir()
    filas = [{"text": f"Resena artificial estrella {star} numero {i}", "label": star - 1}
             for star in range(1,6) for i in range(12)]
    (carpeta / "train.jsonl").write_text("\n".join(json.dumps(x) for x in filas), encoding="utf-8")
    a, total = seleccionar("train", 4, tmp_path, revision)
    b, _ = seleccionar("train", 4, tmp_path, revision)
    assert a == b
    assert total == 60
    assert pd.Series([x["sentimiento"] for x in a]).value_counts().to_dict() == {"negativo":4,"neutro":4,"positivo":4}

def test_cv_registra_baseline_y_particiones_sin_test():
    df = pd.DataFrame({"texto": [f"{word} producto numero {i}" for word in ["excelente", "terrible", "normal"] for i in range(6)],
                       "sentimiento": [label for label in ["positivo","negativo","neutro"] for _ in range(6)]})
    ganador, filas, score = experimentos(df, ["basico"])
    assert len(filas) == 9  # baseline + 2 modelos x 2 ngramas x 2 hiperparametros
    for fila in filas:
        values = [fila[f"fold_{i}"] for i in range(1,4)]
        assert float(np.mean(values)) == pytest.approx(fila["f1_macro_cv"], abs=1e-5)
    assert score == pytest.approx(max(f["f1_macro_cv"] for f in filas), abs=1e-5)

@pytest.mark.parametrize("carpeta", ["artifacts", "artifacts_finales_kelly_entrega"])
def test_pipeline_real_y_backend(carpeta, monkeypatch):
    from src.models import servicio_modelo
    from api import main as api
    ruta = ROOT / carpeta / "pipeline.joblib"
    if not ruta.exists():
        pytest.skip("Requiere ejecutar la reproduccion completa")
    pipeline = joblib.load(ruta)
    textos = ["No me gusto el producto.", "Excelente calidad.", "Es normal, cumple."]
    probs = pipeline.predict_proba(textos)
    assert probs.shape == (3,3)
    assert set(pipeline.classes_) == {"negativo","neutro","positivo"}
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert np.isfinite(probs).all()
    monkeypatch.setattr(servicio_modelo, "RUTA_MODELO", ruta)
    monkeypatch.setattr(api, "MODELO_PATH", ruta)
    servicio_modelo.cargar_pipeline.cache_clear()
    api.cargar_modelo.cache_clear()
    try:
        assert servicio_modelo.predecir_sentimiento(textos[0])["sentimiento"] == pipeline.predict(textos[:1])[0]
        assert api.predecir(api.EntradaTexto(texto=textos[0]))["sentimiento"] == pipeline.predict(textos[:1])[0]
    finally:
        servicio_modelo.cargar_pipeline.cache_clear()
        api.cargar_modelo.cache_clear()


def test_reproduccion_predicciones_historicas():
    ruta = ROOT / "artifacts_finales_kelly_entrega/pipeline.joblib"
    if not ruta.exists():
        pytest.skip("Requiere la ejecucion finales_kelly_entrega")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    historico = joblib.load(ROOT / "artifacts/pipeline.joblib")
    nuevo = joblib.load(ruta)
    np.testing.assert_array_equal(historico.predict(test.texto.tolist()), nuevo.predict(test.texto.tolist()))
    np.testing.assert_allclose(historico.predict_proba(test.texto.tolist()), nuevo.predict_proba(test.texto.tolist()))


def test_no_sobrescribe_pipeline_existente(tmp_path, monkeypatch):
    from src.models import entrenar
    ruta = tmp_path / "pipeline.joblib"
    ruta.write_bytes(b"modelo protegido")
    monkeypatch.setattr("sys.argv", ["entrenar", "--artifacts-dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        entrenar.main()
    assert ruta.read_bytes() == b"modelo protegido"
