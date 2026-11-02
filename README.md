# OpinIA — Análisis de sentimientos con NLP
Proyecto Integrador — Primer Parcial de Tópicos y Programación Avanzada.
Presentación: 3 de noviembre de 2026. Entrega: 2 de noviembre, 23:59; la versión evaluada debe estar en main y marcada entrega-final.

## Integrantes y objetivo
Isaías David: NLP español, API, frontend, integración y pruebas.
Kelly Mendoza: datos, TF-IDF, entrenamiento, experimentos y evaluación.
Aportes verificables en [CONTRIBUCIONES.md](CONTRIBUCIONES.md).
El sistema ayuda a explorar reseñas de productos en español y clasificarlas en negativo, neutro o positivo. Las etiquetas se derivan de estrellas y las predicciones pueden equivocarse.

## Estado actual
Están implementados el NLP compartido, el pipeline entrenado, la API, la predicción con probabilidades y la interfaz con etapas y métricas reales. También existen el informe técnico, las contribuciones y la presentación HTML.
Los resultados comparativos de todos los experimentos, la ejecución guardada del notebook y la verificación desde un entorno limpio siguen pendientes; no deben confundirse con la integración ya terminada.

## Arquitectura
Vue 3 → FastAPI → pipeline joblib (AdaptadorNLP compartido → TF-IDF → clasificador).
La vista NLP invoca /preprocesar y la clasificación /predecir; ambos usan el mismo módulo lingüístico. No se entrena desde la API.
Las métricas se consultan de artifacts/metricas.json; el frontend no fija cifras de modelos anteriores.

## Requisitos y versiones verificadas
Windows/PowerShell, Python 3.13.14 y Node.js para la prueba JavaScript.
Frontend Vue 3.5.13 desde un CDN: requiere conexión a internet.
Dependencias directas observadas fijadas en requirements-isaias.txt, complementario de requirements-kelly.txt; no se modificó el archivo de Kelly.
spaCy 3.8.18 requiere además el paquete de idioma es_core_news_sm 3.8.0. No basta instalar spaCy sin su modelo.
FastAPI 0.162.0, Uvicorn 0.54.0, scikit-learn 1.9.1, NLTK 3.10.4 y demás versiones en el archivo.

## Instalación desde cero: límite de verificación
Se debe obtener el repositorio, crear un entorno virtual .venv con Python compatible, instalar las dependencias y el modelo español. **No se ha ensayado una instalación limpia.**
Para respetar la verificación, no se ofrece una receta no ejecutada como garantizada: quedan pendientes la creación limpia del entorno, la descarga del modelo español y la resolución desde un índice de paquetes.
La comprobación ejecutada en el entorno existente fue:
~~~powershell
.\.venv\Scripts\python.exe -B -m pip install --no-index --disable-pip-version-check -r requirements-isaias.txt
~~~
Todas las dependencias declaradas estaban ya satisfechas. --no-index verifica el entorno/caché local; no instala desde internet en una máquina vacía. Un archivo de versiones directas tampoco es un lock completo de transitivas.

## Datos y entrenamiento: coordinación con Kelly
Se incluyen data/processed/train.csv y test.csv y el modelo serializado: permiten arrancar sin regenerar.
Fuente, fecha y distribución en docs/metadata_dataset.json. 600 train y 180 test; no hay vacíos ni duplicados textuales detectados entre conjuntos. La licencia sigue pendiente de cerrar; revisar las condiciones antes de redistribuir datos.
Los scripts existentes son preparar_datos, eda y entrenar. Se comprobaron estas opciones sin alterar datos ni artefactos:
~~~powershell
.\.venv\Scripts\python.exe -B -m src.models.preparar_datos --help
.\.venv\Scripts\python.exe -B -m src.models.entrenar --help
~~~
El entrenamiento admite --nlp-completo y programa stemming/lematización, unigramas/bigramas y ambos clasificadores. **Pendiente de Kelly:** confirmar el comando integral probado desde datos crudos, entregar resultados/comparacion_modelos.csv, análisis de errores y docs/resultados_experimentales.md. No se publica como verificado un reentrenamiento no ejecutado en esta finalización.
entrenar_revision_kelly.py escribe directorios de revisión; la API usa artifacts/.

## Ejecutar backend y frontend
Desde la raíz del proyecto con el entorno ya preparado:
~~~powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
~~~
Mantener la terminal abierta. No abrir frontend/index.html como archivo local: sus peticiones usan rutas del backend.
- Aplicación: http://127.0.0.1:8000/
- Swagger/API: http://127.0.0.1:8000/docs
- Esquema: http://127.0.0.1:8000/openapi.json
- Salud: http://127.0.0.1:8000/health
- Métricas: http://127.0.0.1:8000/metricas

Si el navegador muestra conexión rechazada, revisar la terminal y confirmar que Uvicorn esté iniciado. El servicio de salud informa existencia del archivo del modelo, no valida su integridad.

## Uso y API
Ingresar entre 3 y 5000 caracteres. Los tres ejemplos solo rellenan el campo; no fuerzan sentimientos.
POST /preprocesar y POST /predecir reciben JSON {"texto":"No me gustaron los productos."}.
El primero devuelve las etapas NLP; el segundo sentimiento y probabilidades. GET /metricas devuelve el JSON real. GET / sirve Vue.
Entradas inválidas → 422; modelo ausente → 503. Documentación con ejemplos, salidas y límites en [informe técnico](docs/informe_tecnico.md#15-api-y-frontend).
El resultado visual distingue sentimientos y presenta todas las probabilidades, sin convertirlas en garantías.

## Pruebas verificadas
~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
node --test tests/test_frontend.cjs
git diff --check
~~~
Python cubre API, validaciones, adaptador NLP, contrato del pipeline y clasificación. JavaScript ejecuta el setup con Vue/fetch simulados y métricas del JSON vigente; no sustituye una prueba visual de navegador.
Se observa advertencia de deprecación de Starlette/httpx. No se cambian dependencias a ciegas para ocultarla.

## Estructura
~~~text
api/                  FastAPI, validación y carga del modelo
frontend/             Vue 3, CSS y panel de métricas
src/nlp/              Procesamiento lingüístico de Isaías
src/models/           Preparación, EDA y entrenamiento de Kelly
data/processed/       Muestras train/test
artifacts/            Pipeline y métricas actuales
notebooks/            Notebook de Kelly
tests/                Pruebas Python y JavaScript
docs/                 Metadatos, informe y presentación
CONTRIBUCIONES.md      Evidencias individuales
requirements-isaias.txt / requirements-kelly.txt
~~~

## Defensa y pendientes
[Presentación HTML](docs/presentacion.html): 12 diapositivas con notas de exposición grupal (~10 min) y guiones individuales (~3 min cada uno). Abrir en el navegador; avanzar con flechas o botones. Imprimir/Guardar como PDF desde el navegador para exportar; no se requiere instalar una herramienta de diapositivas.
Pendientes reales: comparaciones experimentales, ejemplos de errores, EDA/notebook ejecutado, condiciones del dataset, instalación y regeneración limpia, revisión visual y revisión docente.
El panel indica “Comparación experimental pendiente de incorporar” mientras no haya resultados servidos; no inventa experimentos.
El documento oficial es README.md; se consolidó la documentación de las antiguas rutas README.md y Readme.md para evitar su colisión en Windows. q es una captura accidental de diff, conservada sin seguimiento.
Confirmar autorización docente para dos integrantes y su acceso al repositorio. Docker y nube son opcionales.

## Trabajo colaborativo y entrega
main representa la versión estable; se usan ramas por funcionalidad y PR revisados por el compañero. Se conservan feature/nlp-isaias y feature/modelos-kelly como trazabilidad histórica.
Cada integrante registra trabajo desde su cuenta; no compartir cuentas, alterar autores ni reescribir el historial. No se crean commits, merges ni pushes automáticamente.
Después de revisión y validación final, etiquetar el commit evaluado como entrega-final y ensayar un clon de esa etiqueta. La etiqueta todavía no se ha creado en esta sesión.

## Referencias y uso de IA
- [MARC, publicación original](https://aclanthology.org/2020.emnlp-main.369/)
- [Fuente utilizada](https://huggingface.co/datasets/mteb/amazon_reviews_multi)
- [Registro y condiciones originales](https://registry.opendata.aws/amazon-reviews-ml/)
- [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), [FastAPI](https://fastapi.tiangolo.com/), [Vue](https://vuejs.org/)
Repositorio: https://github.com/IsaiasGutierrezTeran/Proyecto_Topicos
Se utilizó ChatGPT/Codex para orientación, código asistido, depuración, pruebas, auditoría y documentación. Los integrantes deben revisar, explicar y modificar lo incorporado. No se atribuyen a la IA verificaciones no ejecutadas.
