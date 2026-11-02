# Resultados experimentales de OpinIA - Kelly Mendoza

## Alcance y auditoria

Se leyo el PDF oficial `C:/Users/kelly/Downloads/Proyecto_Integrador.pdf` (secciones 5.1 a 5.4 y 6 a 10). Exige EDA, comparaciones, ajuste por CV, errores y un comando desde crudos. El minimo de 1.000 registros aparece como ejemplo pendiente de completar, no como umbral cerrado; esta muestra contiene 780.

Rama auditada: `feature/modelos-kelly`, HEAD `beddacb`. Fetch completado. Los aportes de Kelly `ef577bb` y `beddacb` ya estan en `origin/main`, mediante PR #4 y #5; `origin/main` observado: `f6c616c`. El arbol tenia README modificado, instrucciones, guia, resultados previos y un `.bak` sin seguimiento. Se preservaron, sin cambiar rama, commits ni staging. Identidad configurada: Kelly Mendoza <kellymendoza0926@gmail.com>.

Cuando el arbol este limpio, conviene trabajar desde main actualizado en `feature/experimentos-finales-kelly`; no se hizo ese cambio con modificaciones presentes. README, frontend, informe de Isaias y NLP compartido se conservaron.

## Dataset y procedencia verificada

Origen: [Multilingual Amazon Reviews Corpus, Amazon](https://registry.opendata.aws/amazon-reviews-ml/). Autores de la publicacion: Phillip Keung, Yichao Lu, Gyorgy Szarvas y Noah A. Smith. Espejo usado: [MTEB](https://huggingface.co/datasets/mteb/amazon_reviews_multi), subconjunto `es`. Amazon ya no aloja el corpus original. Se verifico el contenido utilizado regenerando la muestra y comparando todas las filas contra los CSV existentes, no solo sus tamanos.

[Condiciones originales](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt): uso limitado para investigacion academica, sin republicacion ni uso comercial; no asociar con datos personales ni identificar autores. No se afirma que el espejo conceda una licencia distinta. Los CSV ya estaban versionados antes de esta tarea: no se agregaron nuevamente ni se publico ningun archivo. Su distribucion previa sigue siendo una brecha que el equipo debe resolver sin reescribir historia.

El registro historico de descarga es `2026-11-02T19:11:27.376265+00:00`, segun metadata local; no es una fecha certificada por el proveedor. Fecha y SHA256 de la nueva obtencion, revision inmutable y hashes procesados constan en el manifiesto y metadata de reproduccion.

Se leyeron 200.000 registros elegibles de train y 5.000 de test del espejo. Muestra: 600 train (200 por clase) y 180 test (60 por clase). Campos del espejo: id, text, label y label_text; los originales documentados incluyen cuerpo, titulo, estrellas, identificadores anonimos y categoria. `text` se conserva como texto de entrada, sin atribuirlo exclusivamente al cuerpo. `label` 0..4 se convierte en estrellas 1..5; 1-2 negativo, 3 neutro, 4-5 positivo. Es supervision indirecta por puntuacion, no anotacion manual de sentimiento.

Muestreo reservoir por estrella sobre cada split oficial completo, semilla 42 para train y 43 para test; estratifica cantidades y no mezcla los splits originales. Se exige texto de al menos ocho caracteres, se recortan espacios y se deduplica por strip/casefold, priorizando train. En la muestra final no hay nulos, duplicados textuales ni solapamiento train/test. Los conteos completos estan en `calidad_datos.json` y `docs/metadata_dataset.json`.

Limitaciones: muestra pequena, balance artificial, dominio comercial, periodo historico y representatividad regional no comprobada. Tres estrellas pueden expresar mezcla de aspectos, no neutralidad linguistica. Los porcentajes no prueban generalizacion fuera de esta muestra.

## EDA ejecutado

`notebooks/eda_kelly.ipynb` tiene celdas ejecutadas, dimension y columnas, nulos, duplicados, clases, longitudes, palabras frecuentes, tres graficas PNG embebidas e interpretaciones. Se explora train; test se consulta para calidad y evaluacion final. Las frecuencias son brutas y no constituyen un segundo normalizador de entrenamiento. La muestra balanceada no requiere pesos de clase ni sobremuestreo.

## Diseno experimental

F1 macro da el mismo peso a negativo, neutro y positivo. Se mantienen los mismos tres folds de `StratifiedKFold(shuffle=True, random_state=42)` en todas las configuraciones y baseline. TF-IDF se ajusta dentro del pipeline en cada fold. Se evaluaron 16 configuraciones, 48 ajustes, mas refits. El cache del adaptador reutiliza solo transformaciones deterministas por texto; no aprende vocabulario fuera de los folds.

MultinomialNB: alpha en {0.5, 1.0}. Regresion logistica: C en {0.5, 1.5}, max_iter=1500, random_state=42; otros valores por defecto de sklearn 1.9.1. Ambos cruzan stemming/lematizacion y ngramas (1,1)/(1,2). Baseline: DummyClassifier(strategy=most_frequent), sin NLP ni TF-IDF. La desviacion es poblacional entre tres folds, no intervalo de confianza. Los resultados de CV tambien tienen optimismo por seleccion; no se uso CV anidada.

### Comparacion de modelos: mejor configuracion de cada familia

| modelo | preprocesamiento | ngramas | hiperparametro | f1_macro_cv | desviacion_cv |
|---|---|---|---|---|---|
| Regresión logística | stemming | unigramas_y_bigramas | 0.5 | 0.59622 | 0.01663 |
| Naive Bayes | stemming | unigramas_y_bigramas | 1.0 | 0.5678 | 0.00876 |
| Mayoritaria | ninguno | no aplica | nan | 0.16541 | 0.0 |

### Comparacion controlada de NLP y ngramas

Regresion logistica con C=0.5 fijo; solo cambia NLP o rango de ngramas.

| modelo | preprocesamiento | ngramas | hiperparametro | f1_macro_cv | desviacion_cv |
|---|---|---|---|---|---|
| Regresión logística | stemming | unigramas_y_bigramas | 0.5 | 0.59622 | 0.01663 |
| Regresión logística | stemming | unigramas | 0.5 | 0.59242 | 0.01212 |
| Regresión logística | lematizacion | unigramas_y_bigramas | 0.5 | 0.58956 | 0.02273 |
| Regresión logística | lematizacion | unigramas | 0.5 | 0.57477 | 0.02309 |

### Tabla completa de configuraciones

| modelo | preprocesamiento | ngramas | hiperparametro | f1_macro_cv | desviacion_cv |
|---|---|---|---|---|---|
| Regresión logística | stemming | unigramas_y_bigramas | 0.5 | 0.59622 | 0.01663 |
| Regresión logística | stemming | unigramas_y_bigramas | 1.5 | 0.59392 | 0.00737 |
| Regresión logística | stemming | unigramas | 0.5 | 0.59242 | 0.01212 |
| Regresión logística | lematizacion | unigramas_y_bigramas | 1.5 | 0.59089 | 0.01179 |
| Regresión logística | stemming | unigramas | 1.5 | 0.59066 | 0.01814 |
| Regresión logística | lematizacion | unigramas_y_bigramas | 0.5 | 0.58956 | 0.02273 |
| Regresión logística | lematizacion | unigramas | 0.5 | 0.57477 | 0.02309 |
| Regresión logística | lematizacion | unigramas | 1.5 | 0.57215 | 0.02279 |
| Naive Bayes | stemming | unigramas_y_bigramas | 1.0 | 0.5678 | 0.00876 |
| Naive Bayes | lematizacion | unigramas_y_bigramas | 1.0 | 0.5649 | 0.00912 |
| Naive Bayes | stemming | unigramas | 1.0 | 0.56418 | 0.01666 |
| Naive Bayes | stemming | unigramas | 0.5 | 0.55931 | 0.02927 |
| Naive Bayes | stemming | unigramas_y_bigramas | 0.5 | 0.55414 | 0.01154 |
| Naive Bayes | lematizacion | unigramas_y_bigramas | 0.5 | 0.55111 | 0.01597 |
| Naive Bayes | lematizacion | unigramas | 0.5 | 0.54725 | 0.00732 |
| Naive Bayes | lematizacion | unigramas | 1.0 | 0.54674 | 0.00986 |
| Mayoritaria | ninguno | no aplica | nan | 0.16541 | 0.0 |

Las tablas CSV conservan los tres puntajes individuales, nombres de configuracion, hiperparametros y todos los parametros TF-IDF. La seleccion se hizo exclusivamente por media de F1 macro CV; test se evalua despues. Stemming y bigramas ganan con C=0.5, pero las diferencias pequenas respecto a otras configuraciones no demuestran significacion estadistica.

## Modelo elegido y TF-IDF

Se conserva regresion logistica, C=0.5, stemming y (1,2). CV=0.59622, desviacion=0.01663. No mejora la referencia historica: la reproduce, por eso no se sustituye el modelo funcional. Clasifica con pesos lineales sobre TF-IDF y probabilidades multiclase; C controla inversamente la regularizacion. Naive Bayes es adecuado para vectores no negativos y textos dispersos, pero su mejor CV (0.56780) es menor. Baseline CV=0.16541, test=0.16667; la ligera diferencia se debe a composicion de folds y desempate de la clase mayoritaria.

TF-IDF real: max_features=5000, vocabulario efectivo=5000, min_df=1, max_df=0.95, sublinear_tf=True, norm=l2, smooth_idf=True, use_idf=True, lowercase=True, strip_accents=unicode, analyzer=word, binary=False; tokenizer y preprocessor=None, stop_words=None porque se filtran en NLP. Token pattern predeterminado: palabras de al menos dos caracteres; dtype float64. Se descartan tokens de una letra en TF-IDF. Los defaults completos quedan en metricas.json.

El adaptador llama exactamente a `src/nlp/preprocesamiento.py`. Conserva negaciones no/ni/nunca/sin/tampoco al filtrar stop words. Normaliza NFC y minusculas, elimina URLs, cifras, simbolos y puntuacion; conserva letras con tildes en NLP y TF-IDF aplica strip_accents. spaCy tokeniza y lematiza; Snowball obtiene raices. No se modifico el normalizador de Isaias. `ejemplos_nlp.json` muestra todas las etapas sobre frases artificiales declaradas como demostracion, no como evidencia de evaluacion.

## Evaluacion final y analisis de errores

Accuracy=0.61667; precision macro=0.60253; recall macro=0.61667; F1 macro=0.60431. Matriz de confusion (filas reales, columnas predichas: negativo, neutro, positivo):

| Real | negativo | neutro | positivo |
|---|---|---|---|
| negativo | 41 | 12 | 7 |
| neutro | 24 | 22 | 14 |
| positivo | 2 | 10 | 48 |

Se aciertan 111/180; 69 errores. Neutral tiene recall 22/60=0.36667 y F1=0.42308: 24 neutros pasan a negativo y 14 a positivo. Probabilidades no se calibraron, por lo que no se interpretan como certeza.

Los archivos locales `predicciones_test_local.csv` y `analisis_ejemplos_local.csv` contienen textos completos, etiquetas, tres probabilidades, aciertos/errores y posibles causas; estan ignorados por Git. Los ejemplos publicables usan indice de fila cero-based del CSV local, sin reproducir texto. Para revision docente local se recupera el texto original por ese indice.

Lectura manual de errores neutros: indice 4 describe rotura y frustracion pese a tres estrellas; indice 8 no ha usado aun el producto y critica el montaje; indice 15 contrapone buen producto y caja deteriorada; indice 22 combina valoracion favorable con manchas y una pieza rota; indice 27 contrapone defectos y buen embalaje. Estos casos respaldan desacuerdo puntuacion/texto o valoraciones mixtas. No hay evidencia suficiente para atribuir ironia. Las otras sugerencias automaticas se rotulan como hipotesis, no causas demostradas.

Ejemplos correctos y errores seleccionados con probabilidades, sin texto del corpus:

| real | predicho | confianza_prediccion | p_negativo | p_neutro | p_positivo | correcto | indice_test |
|---|---|---|---|---|---|---|---|
| neutro | positivo | 0.374 | 0.27204 | 0.35394 | 0.37402 | False | 15 |
| neutro | negativo | 0.3665 | 0.36647 | 0.341 | 0.29253 | False | 22 |
| neutro | negativo | 0.4023 | 0.40225 | 0.29852 | 0.29922 | False | 27 |
| neutro | negativo | 0.4053 | 0.40534 | 0.34619 | 0.24847 | False | 4 |
| neutro | negativo | 0.3926 | 0.39261 | 0.3342 | 0.27319 | False | 8 |
| neutro | neutro | 0.4085 | 0.35195 | 0.40846 | 0.23959 | True | 3 |
| neutro | neutro | 0.3939 | 0.301 | 0.39393 | 0.30508 | True | 29 |
| negativo | positivo | 0.4646 | 0.2356 | 0.29985 | 0.46456 | False | 2 |
| negativo | neutro | 0.3881 | 0.30532 | 0.38809 | 0.30659 | False | 18 |
| negativo | negativo | 0.4253 | 0.42528 | 0.37385 | 0.20086 | True | 5 |
| negativo | negativo | 0.4683 | 0.46828 | 0.32574 | 0.20598 | True | 11 |
| positivo | neutro | 0.4081 | 0.21023 | 0.40805 | 0.38172 | False | 7 |
| positivo | negativo | 0.4561 | 0.45615 | 0.29668 | 0.24717 | False | 9 |
| positivo | positivo | 0.4612 | 0.26012 | 0.27866 | 0.46123 | True | 0 |
| positivo | positivo | 0.4411 | 0.23957 | 0.31936 | 0.44107 | True | 1 |

## Ejemplos antes/despues de NLP

- Original artificial: No me gustaron los productos, aunque la entrega fue rapida.
- Stemming: no gust product entreg rap
- Lematizacion: no gustar producto entrega rapir

- Original artificial: El producto funciona, pero no cumple lo prometido.
- Stemming: product funcion no cumpl promet
- Lematizacion: producto funcionar no cumplir prometer


## Reproducibilidad y proteccion de artefactos

Desde PowerShell, en C:/Proyecto_Topicos, con dependencias instaladas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reproducibles-kelly.txt
.\.venv\Scripts\python.exe -m src.models.reproducir_kelly --revision c379a6705fec24a2493fa68e011692605f44e119 --run-id finales_kelly_entrega
```

El comando se ejecuto realmente para esta evidencia. Para otra ejecucion cambia run-id, por ejemplo `finales_kelly_revision2`, para conservar los anteriores. El prefijo `finales_kelly` mantiene excluidos los modelos y textos privados. El orquestador reutiliza preparar_datos y entrenar; conserva JSONL crudos en data/raw/<revision>, genera muestra separada, exige igualdad con CSV historicos, ejecuta CV, evalua el ganador, serializa, verifica predict/proba tras recarga y ejecuta el notebook. Descarga inicial requiere Internet; posteriores lecturas usan los crudos conservados. Los hashes permiten auditar integridad; no se garantiza que el proveedor mantenga el espejo disponible.

Python observado: 3.15.2. Versiones exactas de todas las dependencias, spaCy y modelo es_core_news_sm 3.8.0 quedan fijadas en requirements-reproducibles-kelly.txt. No se promete compatibilidad de joblib con otras versiones. El backend carga el pipeline; necesita el modulo compartido importable y el recurso spaCy instalado, sin reconstruir reglas ni reentrenar.

Hashes de produccion antes/despues identicos: True. Los artefactos finales son copias aisladas, y no reemplazan artifacts/pipeline.joblib.

Resultados de esta ejecucion: `resultados_finales_kelly_entrega/`. Metricas agregadas y configuraciones son publicables; las rese?as y errores con texto no.

## Archivos entregados

- Modificados: src/models/preparar_datos.py, src/models/entrenar.py, docs/metadata_dataset.json, notebooks/eda_kelly.ipynb y .gitignore.
- Creados: src/models/reproducir_kelly.py, src/models/evidencia_kelly.py, tests/test_experimentos_finales_kelly.py, requirements-reproducibles-kelly.txt y este informe.
- Evidencia: comparacion_modelos.csv, comparacion_nlp.csv, calidad_datos.json, ejemplos_nlp.json, ejemplos_sin_texto.csv, metadata_reproduccion.json, manifest.json, metricas.json.
- Solo locales: crudos, muestra reproducida, pipeline nuevo, predicciones completas y analisis con textos.

## Pruebas

Comando de verificacion:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_modelos_kelly.py tests/test_pipeline_interface.py tests/test_experimentos_finales_kelly.py tests/test_api.py
```

Se verifican etiquetas, NLP compartido y negaciones, reservoir repetible, registro de CV, serializacion, predict, predict_proba, classes_, servicio, API y validacion de entradas. Resultado final guardado en pruebas.txt tras ejecutar la suite sobre los artefactos definitivos. Advertencia de deprecacion Starlette/HTTPX; no fallo funcional. El registro final de pruebas acompana los resultados.

## Brechas y traspaso a Isaias

1. Los CSV historicos ya estan rastreados pese a la prohibicion de republicar. Se debe revisar su distribucion; no se elimino trabajo ajeno ni se publico nada.
2. En esta rama, /metricas del backend tiene el bloque de lectura fuera del endpoint y devuelve None. Es una incidencia previa fuera del alcance de ML; integrar main y revisarla con Isaias. Los contratos de prediccion si se comprobaron.
3. CONTRIBUCIONES.md, diapositivas, README comun, declaracion de IA en el informe comun y tag entrega-final requieren integracion del equipo. Este informe declara asistencia de Codex para codigo, auditoria y experimentos; Kelly debe revisar y poder explicar cada aporte.
4. No se hicieron commits, push ni PR. Se requiere autorizacion de Kelly para commits. La fecha limite oficial es el 2 de noviembre antes de 23:59; main y tag finales quedan a cargo del equipo.
5. No se reclama mejora ni significacion estadistica. La neutralidad requiere etiquetas mejores y mas datos para futuros trabajos, sin reutilizar este test para elegir otro ganador.

## Commits propuestos, solo tras autorizacion

No usar git add . ni subir resultados previos sin inspeccion. Comandos para la rama actual, despues de revisar la diferencia completa y confirmar que no hay staged ajeno:

```powershell
git diff --cached --name-only
git add -- .gitignore src/models/preparar_datos.py docs/metadata_dataset.json notebooks/eda_kelly.ipynb
git diff --cached --name-only
git commit -m "docs(data): documentar procedencia y analisis exploratorio"
git add -- src/models/entrenar.py src/models/reproducir_kelly.py src/models/evidencia_kelly.py requirements-reproducibles-kelly.txt docs/resultados_experimentales.md
git add -- resultados_finales_kelly_entrega/comparacion_modelos.csv resultados_finales_kelly_entrega/comparacion_nlp.csv resultados_finales_kelly_entrega/calidad_datos.json resultados_finales_kelly_entrega/ejemplos_nlp.json resultados_finales_kelly_entrega/ejemplos_sin_texto.csv resultados_finales_kelly_entrega/metadata_reproduccion.json resultados_finales_kelly_entrega/manifest.json resultados_finales_kelly_entrega/metricas.json
git diff --cached --name-only
git commit -m "feat(ml): registrar comparaciones de NLP y modelos"
git add -- tests/test_experimentos_finales_kelly.py resultados_finales_kelly_entrega/pruebas.txt
git diff --cached --name-only
git commit -m "test(ml): verificar reproducibilidad y predicciones"
git push -u origin feature/modelos-kelly
```

Estos comandos no incluyen README modificado previamente, carpetas de revisiones anteriores ni .bak. Si se migra a feature/experimentos-finales-kelly desde main actualizado con arbol limpio, el push debe usar ese nombre. No mover las modificaciones actuales sin preservar y revisar los commits.

## Pull Request sugerido

Titulo: `feat(ml): completar experimentos reproducibles y evidencia de Kelly`

Descripcion: Completa la trazabilidad del dataset con revision y hashes, ejecuta EDA y registra 16 configuraciones NLP/modelo y baseline con los mismos folds. Reproduce la regresion logistica existente (F1 CV 0.59622; F1 test 0.60431), conserva el artefacto funcional y agrega analisis de errores y un comando desde crudos. Incluye pruebas de reproducibilidad, serializacion y carga desde backend. Los textos del corpus quedan locales por sus condiciones de uso. Revisar la distribucion previa de CSV y el endpoint de metricas durante la integracion.

En GitHub: Compare & pull request, base main, compare la rama publicada; pegar esta descripcion y solicitar revision a Isaias mediante su cuenta `IsaiasGutierrezTeran`. Revisar Files changed, confirmar que no hay corpus ni archivos locales, y completar la revision antes de merge. No se abrio ni envio una solicitud automaticamente.
