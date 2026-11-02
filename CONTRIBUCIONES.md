# Contribuciones de OpinIA

Registro basado en el historial Git inspeccionado el 2 de noviembre de 2026.
Los cambios locales sin commit no se presentan como contribuciones ya registradas.

| Integrante | Trabajo acreditado | Archivos | Commits representativos | PR relacionados |
|---|---|---|---|---|
| Isaías David | NLP español: limpieza, tokens, stop words con negaciones, stemming y lematización | `src/nlp/preprocesamiento.py` | `f4ea92d` | #1 |
| Isaías David | API REST, validaciones y pruebas básicas | `api/main.py`, `tests/test_api.py` | `83bd1b5`, `ae93255` | #1 |
| Isaías David | Interfaz Vue e integración; panel de métricas | `frontend/index.html`, `tests/test_frontend.cjs` | `02ff54d`, `0361e9a`, `e57e8e2` | #1, #6 |
| Isaías David | Arquitectura y explicación inicial de NLP | `docs/informe_tecnico.md` | `e3d6228` | #1 |
| Kelly Mendoza | Descarga y preparación del dataset, metadatos, EDA, adaptador NLP, entrenamiento, evaluación y pruebas ML | `src/models/`, `docs/metadata_dataset.json`, `notebooks/eda_kelly.ipynb`, `tests/test_modelos_kelly.py`, `tests/test_pipeline_interface.py`, `requirements-kelly.txt` | `ef577bb` | #4 |
| Kelly Mendoza | Pipeline integrado con NLP compartido, actualización del modelo y métricas, servicio y script de revisión | `artifacts/`, `src/models/entrenar_revision_kelly.py`, `src/models/servicio_modelo.py` | `beddacb` | #5 |

## Evidencia y revisión

Los merges #1 (`781dc4a`), #4 (`85a8c69`), #5 (`956a516`) y #6
(`f6c616c`, referencia origin/main consultada) constan en Git. Un merge no demuestra
por sí solo una aprobación: las revisiones del compañero deben comprobarse en GitHub.
Los revert y restauraciones históricos se conservan; no se atribuyen a nuevos desarrollos.

## Trabajo local pendiente de revisión

La finalización del frontend, documentación, presentación y pruebas ampliadas se
prepara desde el área de Isaías con apoyo de IA. Todavía no tiene commit propio ni
PR nuevo. Este archivo debe actualizarse con sus identificadores reales después
de que el autor revise, comprenda y registre los cambios. Los experimentos nuevos
de Kelly deben registrarse desde su propia cuenta y no se dan por entregados aquí.

Cada integrante debe explicar sus decisiones y el funcionamiento global. No se
crearon commits, fechas, coautorías ni revisiones ficticias.
