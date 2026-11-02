
Proyecto Tópicos — Análisis de Sentimientos con NLP
Materia: Tópicos y Programación Avanzada
Trabajo: Proyecto Integrador — Primer Parcial
Presentación: 3 de noviembre de 2026

Integrantes
Isaías David: normalización, tokenización, stop words, stemming, lematización, integración con API y pruebas.
Kelly Mendoza: vectorización TF-IDF, comparación de n-gramas, entrenamiento, evaluación y experimentos de modelos.
Las contribuciones efectivas se documentarán en CONTRIBUCIONES.md y en el historial de Git. Cada integrante trabaja desde su cuenta y abre pull requests para integrar cambios.

Descripción
Aplicación web para analizar reseñas escritas en español y clasificarlas por sentimiento (positivo, neutro o negativo, según las etiquetas disponibles en el dataset seleccionado). Permitirá ver el resultado, su probabilidad estimada, las etapas de preprocesamiento y métricas de evaluación.

Requisitos académicos
Dataset documentado (origen, licencia, fecha de obtención), análisis exploratorio y división estratificada de entrenamiento/prueba con semilla fija.
Preprocesamiento: normalización, tokenización, stop words en español, stemming y lematización; ejemplos antes/después y justificación.
Comparar stemming frente a lematización, y unigramas frente a unigramas + bigramas.
Vectorización TF-IDF y uno o dos modelos de ML vistos en clase (propuesta: Naive Bayes y regresión logística), con métricas, validación cruzada y análisis de errores.
Pipeline reproducible de entrenamiento; serialización y reutilización del mismo pipeline en la API.
API REST con endpoints de predicción, métricas y salud; validaciones y pruebas automatizadas.
Frontend con entrada de texto, resultado, etapas NLP y métricas.
Documentación y defensa individual, con commits propios verificables.
Tecnologías propuestas
Python, scikit-learn, NLTK / spaCy
FastAPI para la API REST
HTML, CSS y JavaScript (frontend sencillo)
Git y GitHub
Estructura prevista
Proyecto_Topicos/
├── data/               # Dataset o script de descarga, origen y licencia
├── notebooks/          # EDA y experimentos
├── src/
│   ├── nlp/            # Preprocesamiento
│   └── models/         # Entrenamiento y evaluación
├── api/                # API REST
├── frontend/           # Interfaz web
├── tests/              # Pruebas automatizadas
├── docs/               # Informe técnico y presentación
├── README.md
├── CONTRIBUCIONES.md
└── .gitignore
Instalación y ejecución
Estado: en desarrollo. Los comandos exactos de instalación, entrenamiento, pruebas y arranque se completarán y verificarán cuando existan los archivos correspondientes. No se declara todavía el sistema como funcional o desplegado.

Trabajo colaborativo
main: versión estable.
feature/nlp-isaias: funcionalidades de NLP de Isaías.
feature/modelos-kelly: entrenamiento y experimentos de Kelly.
Cada funcionalidad se integra mediante pull request revisado por el otro integrante.
No compartir cuentas, inventar commits ni usar push --force sobre main.
Dataset, resultados y uso de IA
Dataset: pendiente de selección y documentación.
Métricas: pendientes de entrenamiento y evaluación; no se presentan cifras inventadas.
Uso de asistentes de IA: se declarará de forma transparente en el informe técnico; el equipo revisará y podrá explicar y modificar el código.
Entrega
Repositorio: https://github.com/IsaiasGutierrezTeran/Proyecto_Topicos
Fecha límite: 2 de noviembre de 2026, 23:59 (hora de Bolivia).
La versión de entrega se identificará con la etiqueta Git entrega-final.

