
from fastapi.testclient import TestClient
from api.main import app

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
