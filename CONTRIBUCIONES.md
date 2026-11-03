# Contribuciones de OpinIA

Registro basado en el historial Git inspeccionado el 2 de noviembre de 2026.
Los cambios locales sin commit no se presentan como contribuciones ya registradas.

| Integrante | Trabajo acreditado | Archivos | Commits representativos | PR relacionados |
|---|---|---|---|---|
| Isaías David | NLP español: limpieza, tokens, stop words con negaciones, stemming y lematización | `src/nlp/preprocesamiento.py` | `f4ea92d` | #1 |
| Isaías David | API REST, validaciones y pruebas básicas | `api/main.py`, `tests/test_api.py` | `83bd1b5`, `ae93255` | #1 |
| Isaías David | Interfaz Vue e integración; panel de métricas | `frontend/index.html`, `tests/test_frontend.cjs` | `02ff54d`, `0361e9a`, `e57e8e2` | #1, #6 |
| Isaías David | Finalización de interfaz, validaciones, documentación, presentación y pruebas ampliadas | frontend, API, tests, docs, requirements-isaias.txt | `da89321` | #7 |
| Isaías David | Consolidación del README oficial | `README.md`, retirada de la ruta duplicada | `ed1da05` | #8 |
| Kelly Mendoza | Experimentos completos, EDA ejecutado, trazabilidad del dataset, orquestador reproducible y pruebas finales | docs/resultados_experimentales.md, resultados/, resultados_finales_kelly_entrega/, notebook, src/models/, tests/test_experimentos_finales_kelly.py | `0ce0d00` | #11 |
| Isaías David | Arquitectura y explicación inicial de NLP | `docs/informe_tecnico.md` | `e3d6228` | #1 |
| Kelly Mendoza | Descarga y preparación del dataset, metadatos, EDA, adaptador NLP, entrenamiento, evaluación y pruebas ML | `src/models/`, `docs/metadata_dataset.json`, `notebooks/eda_kelly.ipynb`, `tests/test_modelos_kelly.py`, `tests/test_pipeline_interface.py`, `requirements-kelly.txt` | `ef577bb` | #4 |
| Kelly Mendoza | Pipeline integrado con NLP compartido, actualización del modelo y métricas, servicio y script de revisión | `artifacts/`, `src/models/entrenar_revision_kelly.py`, `src/models/servicio_modelo.py` | `beddacb` | #5 |

## Evidencia y revisión

Los merges #1 (`781dc4a`), #4 (`85a8c69`), #5 (`956a516`) y #6
(`f6c616c`, referencia origin/main consultada) constan en Git. Un merge no demuestra
por sí solo una aprobación: las revisiones del compañero deben comprobarse en GitHub.
Los revert y restauraciones históricos se conservan; no se atribuyen a nuevos desarrollos.

## Integración final
El cierre conecta los CSV reales de Kelly con GET /experimentos, visualiza comparaciones controladas, actualiza README, informe y presentación y añade pruebas de API/frontend. Es trabajo del área de integración de Isaías con apoyo de IA, registrado por el commit que contiene esta versión; su identificador se consulta con git log -- api/main.py frontend/index.html.
La publicación en main fue solicitada expresamente por Isaías. No se atribuye este código a Kelly ni sus experimentos a Isaías, y no se afirma revisión de un PR inexistente. Se preservan todos los commits anteriores y las identidades configuradas.
Los merges #7 (8a3aeb0), #8 (21e6e09) y #11 (81687a3) constan en el historial; eso no demuestra por sí solo aprobaciones.

Cada integrante debe explicar sus decisiones y el funcionamiento global. No se
crearon commits, fechas, coautorías ni revisiones ficticias.
