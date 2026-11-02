
from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)


def test_estado_api():
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "activo"


def test_preprocesamiento():
    respuesta = client.post(
        "/preprocesar",
        json={
            "texto": "No me gustó el producto."
        }
    )

    assert respuesta.status_code == 200

    datos = respuesta.json()

    assert "normalizado" in datos
    assert "tokens" in datos
    assert "sin_stopwords" in datos
    assert "stemming" in datos
    assert "lematizacion" in datos

    assert "no" in datos["sin_stopwords"]


def test_texto_vacio():
    respuesta = client.post(
        "/preprocesar",
        json={"texto": ""}
    )

    assert respuesta.status_code == 422


def test_texto_demasiado_largo():
    respuesta = client.post(
        "/preprocesar",
        json={"texto": "a" * 5001}
    )

    assert respuesta.status_code == 422


def test_formato_incorrecto():
    respuesta = client.post(
        "/preprocesar",
        json={"texto": 123}
    )

    assert respuesta.status_code == 422


@pytest.mark.parametrize('ruta', ['/preprocesar', '/predecir'])
@pytest.mark.parametrize('texto', ['', '   ', '123 !!!', 'a' * 5001, 123])
def test_validaciones_de_entrada(ruta, texto):
    respuesta = client.post(ruta, json={'texto': texto})
    assert respuesta.status_code == 422
    assert 'detail' in respuesta.json()


def test_metricas_reales():
    import json
    from api.main import METRICAS_PATH
    with METRICAS_PATH.open(encoding='utf-8') as archivo:
        esperadas = json.load(archivo)
    respuesta = client.get('/metricas')
    assert respuesta.status_code == 200
    assert respuesta.json() == esperadas


def test_prediccion_pipeline_compartido():
    from api.main import cargar_modelo
    texto = 'No me gustaron los productos.'
    modelo = cargar_modelo()
    respuesta = client.post('/predecir', json={'texto': texto})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos['sentimiento'] == str(modelo.predict([texto])[0])
    assert set(datos['probabilidades']) == set(modelo.classes_)
    assert sum(datos['probabilidades'].values()) == pytest.approx(1)
    assert all(0 <= p <= 1 for p in datos['probabilidades'].values())


def test_adaptador_nlp_comparte_preprocesamiento():
    from src.models.adaptador_nlp import AdaptadorNLP
    from src.nlp.preprocesamiento import procesar_texto
    texto = 'No me gustaron los productos.'
    etapas = procesar_texto(texto)
    for metodo in ['stemming', 'lematizacion']:
        assert AdaptadorNLP(metodo=metodo).transform([texto]) == [' '.join(etapas[metodo])]
