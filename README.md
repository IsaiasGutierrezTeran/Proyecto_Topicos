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
El PR #11 de Kelly aporta 16 configuraciones y una línea base, notebook ejecutado, análisis de errores y evidencia de regeneración. La API y el panel integran esos CSV reales. Se comprobó el flujo real en Edge, en escritorio y móvil. La instalación desde un clon limpio aún debe ensayarse; la regeneración registrada por Kelly no se atribuye a una ejecución nueva en esta máquina.

## Arquitectura
Vue 3 → FastAPI → pipeline joblib (AdaptadorNLP compartido → TF-IDF → clasificador).
La vista NLP invoca /preprocesar y la clasificación /predecir; ambos usan el mismo módulo lingüístico. No se entrena desde la API.
Las métricas se consultan de artifacts/metricas.json y GET /experimentos convierte resultados/comparacion_modelos.csv a JSON. El frontend no fija cifras de modelos anteriores ni necesita instalar pandas para servir el CSV.

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

## Datos y entrenamiento reproducible
Fuente y condiciones en docs/metadata_dataset.json y [resultados experimentales de Kelly](docs/resultados_experimentales.md). La muestra tiene 600 train y 180 test, balanceada por sentimiento, sin vacíos ni coincidencias textuales detectadas. Las condiciones originales están documentadas; sigue pendiente revisar la distribución de los CSV históricos ya versionados.

Kelly registró la ejecución desde crudos con revisión inmutable c379a6705fec24a2493fa68e011692605f44e119:
~~~powershell
.\.venv\Scripts\python.exe -m src.models.reproducir_kelly --revision c379a6705fec24a2493fa68e011692605f44e119 --run-id finales_kelly_entrega
~~~
Este comando fue ejecutado por Kelly, según el manifiesto entregado; no se volvió a ejecutar en esta integración. Conserva artifacts/ de producción y escribe resultados/modelos aislados. Para repetir debe elegirse un run-id nuevo que empiece por finales_kelly; no sobrescribir una ejecución previa.
requirements-reproducibles-kelly.txt registra el entorno completo de Kelly (Python 3.15.2) y la URL del modelo español. Esta máquina usa Python 3.13.14 y requirements-isaias.txt: las pruebas de integración pasan, pero eso no demuestra identidad de entornos ni instalación limpia.
La primera descarga requiere internet. Los modelos experimentales y textos completos quedan locales; dos pruebas se omiten aquí porque ese modelo aislado no se versionó.

Entregables: resultados/comparacion_modelos.csv, resultados_finales_kelly_entrega/, docs/resultados_experimentales.md y notebooks/eda_kelly.ipynb con seis celdas ejecutadas. Se preservan los scripts de Kelly, el pipeline productivo y las métricas.

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
- Comparaciones: http://127.0.0.1:8000/experimentos

Si el navegador muestra conexión rechazada, revisar la terminal y confirmar que Uvicorn esté iniciado. El servicio de salud informa existencia del archivo del modelo, no valida su integridad.

## Uso y API
Ingresar entre 3 y 5000 caracteres. Los tres ejemplos solo rellenan el campo; no fuerzan sentimientos.
POST /preprocesar y POST /predecir reciben JSON {"texto":"No me gustaron los productos."}.
El primero devuelve las etapas NLP; el segundo sentimiento y probabilidades. GET /metricas devuelve el JSON real. GET /experimentos devuelve fuente y experimentos, con modelo, NLP, n-gramas, hiperparámetro, media/desviación y los tres folds. Si el CSV falta, está vacío o contiene datos inválidos, responde 503 sin bloquear las métricas. GET / sirve Vue.
Entradas inválidas → 422; modelo ausente → 503. Documentación con ejemplos, salidas y límites en [informe técnico](docs/informe_tecnico.md#15-api-y-frontend).
El resultado visual distingue sentimientos y presenta todas las probabilidades, sin convertirlas en garantías.

## Pruebas verificadas
~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
node --test tests/test_frontend.cjs
git diff --check
~~~
Python cubre API, validaciones, adaptador NLP, contrato del pipeline y clasificación. JavaScript ejecuta el setup con Vue/fetch simulados y métricas del JSON vigente; no sustituye una prueba visual de navegador.
Resultado de esta integración: 35 pruebas Python aprobadas, 2 omitidas por falta del modelo experimental aislado y 1 advertencia Starlette/httpx; prueba JavaScript aprobada. En Edge real se verificaron predicción, seis etapas NLP, cuatro indicadores y tablas de 3, 4 y 17 resultados, sin excepciones JavaScript ni desbordamiento horizontal de la página en 1440 y 390 píxeles. Las tablas móviles se desplazan dentro de su contenedor. El registro de Kelly contiene 17 aprobadas en su entorno: es evidencia distinta, no una ejecución nueva aquí.

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
docs/                 Metadatos, informe, experimentos y presentación
resultados/           CSV comparativo servido por la API
resultados_finales_kelly_entrega/  Evidencia reproducible de Kelly
CONTRIBUCIONES.md      Evidencias individuales
requirements-isaias.txt / requirements-kelly.txt / requirements-reproducibles-kelly.txt
~~~

## Defensa y pendientes
[Presentación HTML](docs/presentacion.html): 12 diapositivas con notas de exposición grupal (~10 min) y guiones individuales (~3 min cada uno). Abrir en el navegador; avanzar con flechas o botones. Imprimir/Guardar como PDF desde el navegador para exportar; no se requiere instalar una herramienta de diapositivas.
El panel incluye un botón Consultar comparaciones: muestra la mejor configuración por familia y una comparación controlada de stemming/lematización y n-gramas, manteniendo fijos modelo y parámetro del ganador. Permite desplegar los 17 experimentos y sus folds. Puntajes de CV y test se identifican por separado; la desviación se expresa en puntos porcentuales.
Pendientes reales: ensayo de instalación desde un clon limpio, preparación de la defensa, comprobación docente y decisión sobre la distribución de los CSV históricos según sus condiciones. Los experimentos y el EDA ya están entregados.
El documento oficial es README.md; se consolidó la documentación de las antiguas rutas README.md y Readme.md para evitar su colisión en Windows. q es una captura accidental de diff, conservada sin seguimiento.
Confirmar autorización docente para dos integrantes y su acceso al repositorio. Docker y nube son opcionales.

## Trabajo colaborativo y entrega
main representa la versión estable; se usan ramas por funcionalidad y PR revisados por el compañero. Se conservan feature/nlp-isaias y feature/modelos-kelly como trazabilidad histórica.
Cada integrante registra trabajo desde su cuenta; no compartir cuentas, alterar autores ni reescribir el historial. La publicación de esta integración en main fue solicitada expresamente por Isaías; eso no equivale a una revisión del compañero.
La versión de entrega se identifica con entrega-final sobre el commit publicado en main. Antes de la defensa debe ensayarse un clon de esa etiqueta; la existencia del tag no certifica por sí sola todos los criterios de la rúbrica.

## Referencias y uso de IA
- [MARC, publicación original](https://aclanthology.org/2020.emnlp-main.369/)
- [Fuente utilizada](https://huggingface.co/datasets/mteb/amazon_reviews_multi)
- [Registro y condiciones originales](https://registry.opendata.aws/amazon-reviews-ml/)
- [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), [scikit-learn](https://scikit-learn.org/), [FastAPI](https://fastapi.tiangolo.com/), [Vue](https://vuejs.org/)
Repositorio: https://github.com/IsaiasGutierrezTeran/Proyecto_Topicos
Se utilizó ChatGPT/Codex para orientación, código asistido, depuración, pruebas, auditoría y documentación. Los integrantes deben revisar, explicar y modificar lo incorporado. No se atribuyen a la IA verificaciones no ejecutadas.
