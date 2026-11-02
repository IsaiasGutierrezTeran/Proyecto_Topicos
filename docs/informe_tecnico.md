# Informe técnico — OpinIA
Fecha de actualización: 2 de noviembre de 2026. Estado: integración verificada; comparaciones experimentales pendientes de entrega por Kelly.

## 1. Resumen
OpinIA clasifica reseñas españolas como negativas, neutras o positivas. Integra procesamiento lingüístico, TF-IDF, un clasificador serializado, FastAPI y Vue 3. Permite inspeccionar el texto y las métricas. Los resultados son estimaciones estadísticas, no certezas sobre la intención del autor.

## 2. Problema
La lectura manual de reseñas dificulta identificar tendencias. Una única puntuación puede ocultar opiniones mixtas, negaciones o sarcasmo. El sistema facilita exploración y clasificación sin sustituir el juicio humano.

## 3. Justificación
La clasificación supervisada corresponde a etiquetas disponibles derivadas de estrellas. Naive Bayes y regresión logística son adecuados para comparar modelos de clasificación de texto con vectores dispersos; no se afirma superioridad experimental sin la tabla completa. FastAPI aporta validación y OpenAPI; Vue permite consumir la API sin otra aplicación de servidor.

## 4. Objetivos
Construir y justificar NLP español; entrenar y evaluar clasificadores; comparar stemming/lematización y n-gramas; reutilizar el pipeline en producción; exponer resultados interpretables y conservar trazabilidad individual.

## 5. Dataset y procedencia
Fuente: [mteb/amazon_reviews_multi, español](https://huggingface.co/datasets/mteb/amazon_reviews_multi), espejo del [Multilingual Amazon Reviews Corpus (MARC)](https://www.amazon.science/publications/the-multilingual-amazon-reviews-corpus).
Metadatos: `docs/metadata_dataset.json`; descarga registrada el 2026-11-02 a las 19:11:27 UTC. Se usan 600 ejemplos train y 180 test, con 200 y 60 por sentimiento respectivamente. Orígenes declarados: 200.000 train y 5.000 test. Etiquetas: 1–2 estrellas negativo; 3 neutro; 4–5 positivo.
La auditoría contrastó las 780 muestras con los splits fuente, normalizando espacios y saltos de línea; coincidieron texto, estrellas y sentimiento. No se interpretan las estrellas como una anotación lingüística perfecta.
El muestreo reservoir por estrella usa semilla 42 y conserva la separación original; no utiliza train_test_split. La deduplicación puede alterar cuotas y debe revisarse al regenerar.
**Pendiente de Kelly:** cerrar condiciones de uso y redistribución. El metadato todavía dice licencia no verificada. El [registro AWS](https://registry.opendata.aws/amazon-reviews-ml/) informa del cese de alojamiento original y enlaza las [condiciones oficiales](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt). No se atribuye licencia abierta al espejo.

## 6. Análisis exploratorio
Comprobaciones de auditoría: ningún texto vacío ni duplicado textual interno; ninguna coincidencia textual entre train y test. Longitudes en palabras: train media 31,48, mínimo 3, máximo 203; test media 31,26, mínimo 6, máximo 279. No se comprobó independencia por usuario/producto ni duplicación semántica.
`src/models/eda.py` contempla clases, faltantes, longitudes y frecuencias; el notebook existente no guarda ejecuciones ni salidas. **Pendiente:** Kelly debe entregar frecuencias, gráficos y conclusiones reales, preferentemente usando train para decisiones de modelado. Los valores anteriores son comprobaciones de auditoría, no salidas ficticias del notebook.

## 7. Pipeline NLP: implementación, ejemplos y decisiones
Código compartido: `src/nlp/preprocesamiento.py`. spaCy 3.8.18, modelo español es_core_news_sm 3.8.0 y SnowballStemmer de NLTK 3.10.4. Ejemplos obtenidos ejecutando ese código con Unicode explícito.

Entrada real: **¡NO me gustaron los productos! La atención fue pésima 😡. Visita https://ejemplo.com 123**

| Etapa | Salida real |
|---|---|
| Normalización | no me gustaron los productos la atención fue pésima visita |
| Tokens | no, me, gustaron, los, productos, la, atención, fue, pésima, visita |
| Tokens conservados | no, gustaron, productos, atención, pésima, visita |
| Stop words eliminadas, derivadas de las listas | me, los, la, fue |
| Stemming | no, gust, product, atencion, pesim, visit |
| Lematización | no, gustar, producto, atención, pésimo, visita |

**Normalización:** Unicode NFC unifica representaciones y conserva tildes; minúsculas reducen variantes. Se eliminan URLs, dígitos, puntuación y símbolos Unicode, incluidos emojis. Se sustituyen guiones y guiones bajos por espacios y se compactan espacios. Esto simplifica el vocabulario, pero pierde intensidad, puntuaciones y señales emocionales; no se afirma que conservar emojis sea imposible o innecesario. La eliminación de tildes adicional ocurre en TF-IDF mediante strip_accents="unicode"; debe distinguirse del normalizador.

**Tokenización:** spaCy segmenta y solo se muestran tokens is_alpha. Caso difícil ejecutado: "No está mal: atención al cliente, wifi y calidad-precio." produce tokens "no, está, mal, atención, al, cliente, wifi, y, calidad, precio". El guion se pierde; "atención al cliente" no es un token único. Los bigramas se forman después del filtrado, por lo que no representan siempre palabras contiguas del texto original.

**Stop words:** lista española de spaCy menos {no, ni, nunca, sin, tampoco}. Preservar negaciones evita eliminar una señal importante del sentimiento. No se agregan otras palabras en el código actual. En el caso difícil, los conservados son "no, atención, cliente, wifi, calidad, precio": la lista elimina "mal". Es una limitación real para sentimiento que debe evaluarse, no corregirse silenciosamente sin reentrenar.

**Stemming:** Snowball español aplica reglas morfológicas; sus raíces no son necesariamente palabras. "gustaron → gust" y "productos → product". Reduce variantes con costo de pérdida lingüística.
**Lematización:** spaCy estima formas base según análisis lingüístico: "gustaron → gustar", "pésima → pésimo". Puede cometer errores y depende de la versión del modelo. Parser y NER se desactivan; el módulo calcula ambas variantes y el adaptador selecciona una.
**N-gramas/TF-IDF:** el pipeline evalúa (1,1) y (1,2); max_features=5000, min_df=1, max_df=0.95, sublinear_tf=True, lowercase=True y strip_accents="unicode". 5000 es un límite, no el tamaño observado del vocabulario. TF-IDF se ajusta dentro del pipeline de cada fold para evitar fuga desde validación.

## 8. Comparación de técnicas NLP
El entrenamiento define stemming y lematización con --nlp-completo. El artefacto seleccionado usa stemming y (1,2). Esto **no sustituye** la comparación de todas las configuraciones.
Pendiente de integrar `docs/resultados_experimentales.md` cuando Kelly lo entregue y los resultados originales de CV; no existe ese archivo al cierre de esta revisión.

## 9. Modelos y justificación
MultinomialNB trabaja con características no negativas de frecuencia/TF-IDF y supone independencia condicional. LogisticRegression aprende pesos lineales por clase y estima probabilidades; C controla inversamente la regularización. La grilla contiene alpha=[0.5,1.0] y C=[0.5,1.5]. El modelo serializado es regresión logística con C=0.5.

## 10. Tabla comparativa de modelos
| Modelo | Resultado conservado | Evidencia faltante |
|---|---|---|
| Naive Bayes | Sin resultado comparativo disponible | Media/desviación F1 CV, NLP, n-gramas e hiperparámetros |
| Regresión logística | Ganador registrado; F1 macro test 0,60431 | Resultados de cada configuración CV |
| Línea base mayoritaria | F1 macro test registrado 0,16667 | Interpretación completa con los experimentos |

No se rellenan celdas con métricas supuestas. Formato ya producido por el script de Kelly: modelo, preprocesamiento, ngramas, hiperparametro, f1_macro_cv, desviacion_cv en resultados/comparacion_modelos.csv. No se cambió ese esquema. La API actual no sirve ese CSV.

## 11. Validación cruzada y selección
StratifiedKFold con 3 folds, shuffle y semilla 42; GridSearchCV con scoring=f1_macro. Selección usando train; test se reserva para evaluar el ganador. Se registró F1 CV=0,59622, no recalculado en esta finalización. Sin las salidas completas no se certifica la comparación histórica de todas las configuraciones.

## 12. Métricas y matriz de confusión
Instantánea de `artifacts/metricas.json` comprobada contra el modelo y test actuales; revisar si Kelly actualiza los artefactos.

| Accuracy test | Precision macro test | Recall macro test | F1 macro test |
|---|---|---|---|
| 0,61667 | 0,60253 | 0,61667 | 0,60431 |

| Real / Predicho | Negativo | Neutro | Positivo |
|---|---:|---:|---:|
| Negativo | 41 | 12 | 7 |
| Neutro | 24 | 22 | 14 |
| Positivo | 2 | 10 | 48 |

Filas reales, columnas predichas; 111 aciertos y 69 errores sobre 180. Support=60 por clase. El frontend consulta la API: estos números documentales no son constantes de interfaz.

## 13. Análisis de errores
La clase neutra tiene recall 22/60=36,67 %; 24 neutras fueron negativas y 14 positivas. Es una observación cuantitativa reproducible, no una explicación causal. Posibles dificultades: etiquetas por estrellas, opiniones mixtas, negaciones y palabras sentimentales filtradas.
**Pendiente de Kelly:** ejemplos reales de errores y explicación individual. El script exporta resultados/analisis_errores.csv, pero no se encontró esa salida; no se presentan ejemplos inventados como errores del test.

## 14. Arquitectura y flujo
Navegador Vue → POST /preprocesar para inspección → POST /predecir con la misma entrada → pipeline joblib (NLP compartido → TF-IDF → clasificador) → clase/probabilidades → interfaz. GET /metricas carga JSON del disco. El backend no vuelve a implementar la limpieza ni entrena al recibir peticiones. La vista NLP usa el mismo módulo, aunque la API ejecuta sus pasos de inspección y predicción por separado.

## 15. API y frontend
Entrada POST JSON: {"texto":"No me gustaron los productos."}; texto str, 3–5000 caracteres. Ejemplos completos se prueban en /docs.

| Método y URL | Entrada | Salida | Errores |
|---|---|---|---|
| GET / | Ninguna | HTML de OpinIA | Error de archivo si falta el frontend |
| GET /health | Ninguna | {"estado":"activo","modelo_disponible":true} cuando existe el archivo | No comprueba carga ni compatibilidad del modelo |
| POST /preprocesar | JSON texto | original, normalizado, tokens, sin_stopwords, stemming, lematizacion | 422: vacío, longitud, tipo o ausencia de palabras válidas |
| POST /predecir | JSON texto | sentimiento y probabilidades por clase | 422: entrada inválida; 503: modelo ausente |
| GET /metricas | Ninguna | Objeto real de artifacts/metricas.json | 503: archivo ausente o JSON null |
| GET /docs, /openapi.json | Ninguna | Swagger UI y esquema | — |

Para la entrada larga de la sección 7, /preprocesar devuelve exactamente las etapas mostradas. /predecir devolvió {"sentimiento":"negativo","probabilidades":{"negativo":0.45647388768940783,"neutro":0.289522664305396,"positivo":0.25400344800519603}} con los artefactos auditados. No es una salida garantizada para otros modelos.
Ejemplo de petición desde el cliente: fetch('/predecir', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({texto:'No me gustaron los productos.'})}).
Error semántico ejecutado: {"texto":"123 !!!"} → HTTP 422 con detail="El texto no contiene palabras válidas.". Los errores de validación Pydantic tienen detail como lista; el frontend presenta una explicación legible.
Se corrigió un 500 real en /predecir para ese caso capturando ValueError/TypeError; no se alteró el modelo ni las probabilidades. Archivos corruptos/incompatibles y JSON de métricas malformado todavía pueden producir 500.
La interfaz tiene contador, ejemplos editables, carga, errores, colores por sentimiento, seis etapas NLP y métricas accesibles con tablas desplazables. Las stop words eliminadas son una diferencia por ocurrencias entre listas reales. La comparación experimental permanece pendiente.

## 16. Pruebas
Se conservan pruebas existentes y se amplían validaciones de ambos POST, métricas contra JSON, predicción con pipeline real y equivalencia del adaptador para ambos métodos NLP. JavaScript evalúa datos reales de métricas, formato español, carga, errores, entradas inválidas y flujo de reseñas con fetch simulado.
Comandos de validación: .\\.venv\\Scripts\\python.exe -m pytest tests/ -v y node --test tests/test_frontend.cjs.
Resultado ejecutado el 2 de noviembre de 2026: 23 pruebas Python aprobadas y una prueba de archivo JavaScript aprobada (con múltiples aserciones). La advertencia Starlette/httpx observada no se oculta. TestClient no equivale a probar un navegador; la revisión visual sigue pendiente.

## 17. Reproducibilidad
Consultar README. Python observado 3.13.14. requirements-isaias.txt fija dependencias directas observadas, sin cambiar requirements-kelly.txt. es_core_news_sm 3.8.0 es un paquete adicional; no se ensayó su instalación desde cero ni se fijaron todas las dependencias transitivas.
Los scripts de Kelly permiten preparar datos, EDA y entrenamiento; se verificaron las opciones --help, no se reentrenó ni se sobrescribieron datos. La regeneración integral con un comando queda pendiente de coordinación con Kelly. entrenar_revision_kelly.py escribe directorios de revisión; no confundirlos con los usados por la API.
La prueba de arranque utiliza python -m uvicorn api.main:app --host 127.0.0.1 --port 8000. Vue 3.5.13 se carga desde un CDN y necesita conectividad.

## 18. Resultados
La aplicación carga el modelo real, devuelve probabilidades y métricas coherentes con test y expone sus etapas. El pipeline utiliza verdaderamente NLP compartido. No se declara completada la comparación experimental por la sola existencia de su código.

## 19. Limitaciones
Muestra reducida y balanceada artificialmente; sentimiento aproximado por estrellas; posible sarcasmo/mezcla de opiniones; pérdida de emojis e información numérica; stop words con términos sentimentales; licenciamiento/redistribución sin cerrar; instalación limpia, resultados comparativos y notebook ejecutado pendientes.
Git registra README.md y Readme.md: colisión en Windows pendiente de resolver con autorización. El archivo q se conserva sin seguimiento. Confirmar autorización docente para grupo de dos y acceso como colaborador.

## 20. Conclusiones
La integración NLP–ML–API–frontend está operativa y verificable. Las mejoras de UX no cambian el modelo. La entrega académica requiere aún conservar resultados de experimentación y EDA, cerrar condiciones del dataset y verificar instalación y regeneración en limpio. No se puede certificar cumplimiento total antes de completar esos puntos.

## 21. Bibliografía y atribuciones
- Keung, Lu, Szarvas y Smith (2020), The Multilingual Amazon Reviews Corpus: https://aclanthology.org/2020.emnlp-main.369/
- Fuente espejo y condiciones: enlaces de la sección 5.
- spaCy: https://spacy.io/usage/linguistic-features
- NLTK Snowball: https://www.nltk.org/api/nltk.stem.snowball.html
- scikit-learn: https://scikit-learn.org/stable/modules/feature_extraction.html y https://scikit-learn.org/stable/modules/grid_search.html
- FastAPI: https://fastapi.tiangolo.com/ ; Vue: https://vuejs.org/
- Rúbrica oficial: texto de “Proyecto Integrador – Primer Parcial” proporcionado por el docente y compartido en la sesión; PDF no incorporado al repositorio.

## 22. Integrantes y uso de IA
Isaías David: NLP, backend, frontend, integración y documentación. Kelly Mendoza: dataset, adaptador, experimentación, entrenamiento y evaluación. Evidencia individual en CONTRIBUCIONES.md.
Se usó ChatGPT/Codex para orientación, propuestas, implementación asistida, auditoría, pruebas y redacción. Los integrantes deben revisar, entender y poder modificar el código y comprobar todas las afirmaciones. Los cambios locales aún no tienen autoría registrada en un nuevo commit.
