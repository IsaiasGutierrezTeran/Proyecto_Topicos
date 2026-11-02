
# INFORME TÉCNICO
## Sistema Web de Análisis de Sentimientos — OpinIA

### 1. Introducción

OpinIA es una aplicación web desarrollada para analizar
reseñas escritas en español mediante técnicas de
Procesamiento de Lenguaje Natural (NLP) y algoritmos
de Machine Learning.

El sistema permite ingresar un texto, visualizar sus
etapas de procesamiento y obtener una clasificación
de sentimiento a partir de un modelo entrenado.

### 2. Planteamiento del problema

El análisis manual de grandes cantidades de reseñas
requiere tiempo y dificulta identificar patrones en
las opiniones de los usuarios.

El proyecto propone automatizar la clasificación de
sentimientos mediante técnicas de NLP y aprendizaje
automático.

### 3. Objetivo general

Desarrollar una aplicación web que permita analizar
y clasificar sentimientos expresados en reseñas en
español, integrando NLP, Machine Learning, una API
REST y una interfaz web.

### 4. Objetivos específicos

- Implementar técnicas de preprocesamiento NLP.
- Entrenar y evaluar modelos de clasificación.
- Comparar stemming y lematización.
- Comparar unigramas y bigramas mediante TF-IDF.
- Desarrollar una API REST utilizando FastAPI.
- Implementar una interfaz web interactiva.
- Evaluar el funcionamiento mediante pruebas.

### 5. Arquitectura del sistema

El proyecto está organizado en cinco capas:

1. Datos: reseñas y etiquetas de sentimiento.
2. NLP: procesamiento lingüístico del texto.
3. Machine Learning: entrenamiento y clasificación.
4. Backend: API REST desarrollada con FastAPI.
5. Frontend: interfaz web desarrollada con Vue.

### 6. Procesamiento de Lenguaje Natural

El módulo NLP está ubicado en:

src/nlp/preprocesamiento.py

Incluye las siguientes operaciones:

**Normalización:** conversión a minúsculas,
tratamiento de URLs, números, puntuación y símbolos.

**Tokenización:** separación del texto en palabras.

**Stop words:** filtrado de palabras frecuentes,
conservando negaciones relevantes como "no".

**Stemming:** reducción de palabras a raíces
aproximadas mediante SnowballStemmer.

**Lematización:** obtención de formas base utilizando
el modelo lingüístico de spaCy para español.

La comparación experimental entre stemming y
lematización está pendiente de los resultados
del entrenamiento.

### 7. Modelos de Machine Learning

Modelos previstos:

- Naive Bayes.
- Regresión logística.

Vectorización propuesta:

- TF-IDF con unigramas.
- TF-IDF con unigramas y bigramas.

Pendiente: documentar hiperparámetros, validación
cruzada, métricas, análisis de errores y modelo
seleccionado utilizando resultados reales.

### 8. Backend

El backend se desarrolla utilizando FastAPI.

Endpoints:

- GET /health
- POST /preprocesar
- POST /predecir
- GET /metricas

El endpoint de preprocesamiento ya ha sido probado.
La predicción y las métricas dependen de integrar
el modelo entrenado.

### 9. Frontend

La interfaz OpinIA permite:

- Ingresar una reseña.
- Enviar el texto al backend.
- Visualizar las etapas del NLP.
- Consultar la predicción cuando el modelo esté listo.
- Consultar métricas cuando estén disponibles.

### 10. Pruebas

Se implementaron cinco pruebas automatizadas
para verificar:

- Estado de la API.
- Preprocesamiento.
- Rechazo de texto vacío.
- Rechazo de texto excesivamente largo.
- Rechazo de formato incorrecto.

Resultado comunicado: cinco pruebas aprobadas
en el entorno local de Isaías.

### 11. Dataset y resultados

Pendiente de completar con el dataset real,
su fuente, licencia, fecha de obtención,
cantidad de registros, distribución de clases,
métricas y resultados experimentales.

### 12. Conclusiones preliminares

Se ha implementado una arquitectura que comunica
la interfaz web con el backend y permite ejecutar
las etapas del procesamiento NLP.

La clasificación de sentimientos y la evaluación
integral estarán completas cuando se integren
y verifiquen los modelos entrenados.

### 13. Uso de herramientas de inteligencia artificial

Durante el desarrollo se utilizó ChatGPT como
herramienta de apoyo para orientación técnica,
generación de propuestas de código, resolución
de errores y elaboración de documentación.

Los integrantes deben revisar, probar y comprender
el código que incorporen al proyecto.

### 14. Integrantes

Isaías David:
Procesamiento NLP, backend, interfaz e integración.

Kelly Mendoza:
Dataset, experimentación, entrenamiento y evaluación
de modelos de Machine Learning.

Las contribuciones definitivas deben coincidir
con el historial de commits del repositorio.
