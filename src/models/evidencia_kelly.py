"""Evidencia real, agregados publicables y ejemplos completos exclusivamente locales."""
import json
import re
import sys
from pathlib import Path
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from src.nlp.preprocesamiento import procesar_texto

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "resultados_finales_kelly"

def tabla(df):
    cols = list(df.columns)
    return "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n" + "\n".join(
        "| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in row) + " |" for row in df.itertuples(index=False, name=None))

def causa(texto):
    # Hipotesis observables, no anotaciones causales demostradas.
    if re.search(r"\b(pero|aunque|sin embargo)\b", texto, re.I):
        return "Conector adversativo presente; revisar si contrapone valoraciones. Hipotesis, no causa demostrada."
    if re.search(r"\b(no|nunca|tampoco|ni)\b", texto, re.I):
        return "Negacion explicita presente; su alcance puede ser ambiguo. Hipotesis, no causa demostrada."
    if len(texto.split()) < 12:
        return "Menos de 12 palabras: contexto escaso. Hipotesis, no causa demostrada."
    return "Sin causa atribuible automaticamente; requiere lectura manual. No se afirma ironia."

def generar_notebook():
    nb = nbf.v4.new_notebook()
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    nb.cells = [md("# EDA de OpinIA - Kelly Mendoza\nDatos reales de entrenamiento; test queda fuera de decisiones exploratorias. La etiqueta deriva de estrellas, no de anotacion linguistica. No se reproducen resenas completas por las condiciones del corpus."),
        code("from pathlib import Path\nfrom collections import Counter\nimport re\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd()\ndf = pd.read_csv(ROOT / 'data/processed/train.csv')\nprint('Dimension:', df.shape)\nprint('Columnas:', df.columns.tolist())\nprint('Nulos:', df.isna().sum().to_dict())\nprint('Duplicados normalizados:', df.texto.str.strip().str.casefold().duplicated().sum())\nprint('Clases:', df.sentimiento.value_counts().to_dict())"),
        code("fig, ax = plt.subplots(figsize=(7,4))\ndf.sentimiento.value_counts().reindex(['negativo','neutro','positivo']).plot.bar(ax=ax, rot=0)\nax.set(title='Distribucion de entrenamiento', ylabel='Resenas')\nplt.tight_layout()\nplt.show()"),
        code("longitudes = df.texto.str.split().str.len()\nprint(longitudes.describe().round(2).to_string())\nprint(df.assign(palabras=longitudes).groupby('sentimiento').palabras.agg(['mean','median','min','max']).round(2).to_string())\nfig, ax = plt.subplots(figsize=(7,4))\nlongitudes.hist(bins=25, ax=ax)\nax.set(title='Longitud de entrenamiento', xlabel='Palabras', ylabel='Frecuencia')\nplt.tight_layout()\nplt.show()"),
        code("tokens = Counter(w for t in df.texto for w in re.findall(r'\\b[^\\W\\d_]{3,}\\b', t.lower()))\nprint(tokens.most_common(20))\nfig, ax = plt.subplots(figsize=(8,5))\nf = tokens.most_common(15)[::-1]\nax.barh([w for w,n in f], [n for w,n in f])\nax.set(title='Frecuencias brutas, sin quitar stop words', xlabel='Frecuencia')\nplt.tight_layout()\nplt.show()"),
        code("print('Interpretacion: muestra balanceada por diseno; no necesita remuestreo adicional.')\nprint('Longitud tipica (mediana):', longitudes.median(), 'palabras; media:', round(longitudes.mean(),2))\nprint('Las palabras funcionales frecuentes justifican comparar el filtrado compartido de NLP.')\nprint('Las tres estrellas aproximan neutralidad: pueden contener valoraciones mixtas.')\nprint('Bigramas pueden representar negaciones locales; su aporte se decide solo por CV.')"),
        md("## Comparaciones de NLP y modelos\nMismos tres folds estratificados y semilla 42. Resultados de validacion, sin seleccionar por test."),
        code("resultados = pd.read_csv(ROOT / 'resultados_finales_kelly/comparacion_modelos.csv')\nprint(resultados[['modelo','preprocesamiento','ngramas','hiperparametro','f1_macro_cv','desviacion_cv']].to_string(index=False))")]
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("resultados_finales_kelly/", OUT.name + "/")
    nb.metadata.kernelspec = {"display_name": "Python OpinIA", "language": "python", "name": "python3"}
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(nb, km=km, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.validate(nb)
    nbf.write(nb, ROOT / "notebooks/eda_kelly.ipynb")

def generar(out=None):
    global OUT
    if out is not None:
        OUT = Path(out)
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    quality = {}
    for name, frame in [("train", train), ("test", test)]:
        quality[name] = {"filas": len(frame), "nulos": frame.isna().sum().to_dict(),
            "duplicados_texto": int(frame.texto.str.strip().str.casefold().duplicated().sum()),
            "clases": frame.sentimiento.value_counts().to_dict(),
            "estrellas": {str(k): int(v) for k,v in frame.estrellas.value_counts().items()}}
    quality["solapamiento_train_test"] = len(set(train.texto.str.strip().str.casefold()) & set(test.texto.str.strip().str.casefold()))
    (OUT / "calidad_datos.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = ROOT / "docs/metadata_dataset.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta.update({"organizacion_original": "Amazon", "espejo": "Massive Text Embedding Benchmark (MTEB)",
        "autores_publicacion": ["Phillip Keung", "Yichao Lu", "Gyorgy Szarvas", "Noah A. Smith"],
        "fuente_original": "https://registry.opendata.aws/amazon-reviews-ml/",
        "licencia": "Condiciones originales verificadas: uso academico; no republicar ni uso comercial",
        "licencia_url": "https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt",
        "fecha_verificacion": "2026-11-02",
        "columnas_espejo": ["id", "text", "label", "label_text"],
        "columnas_originales_documentadas": ["review text", "review title", "star rating", "anonymized reviewer ID", "anonymized product ID", "product category"],
        "fecha_obtencion_nota": "fecha_descarga_utc es el registro historico local, no una fecha demostrada por el proveedor; nueva reproduccion en resultados_finales_kelly/metadata_reproduccion.json",
        "limpieza": ["texto.strip; minimo 8 caracteres", "reservoir por estrella con semilla 42/43 y splits originales", "eliminar duplicados por strip/casefold dentro y entre splits, priorizando train"],
        "limitaciones": ["Tres estrellas no equivale necesariamente a neutralidad", "Solo dominio de compras; no generaliza a redes sociales ni variantes regionales", "Muestra pequena y balanceada artificialmente", "Origen original retirado de AWS; espejo de terceros", "No hay anotacion manual de sentimiento"],
        "verificacion_muestra": "Regenerada desde JSONL versionados y comparada fila a fila con CSV locales",
        "calidad_observada": quality,
        "reproduccion": json.loads((OUT / "metadata_reproduccion.json").read_text(encoding="utf-8"))})
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = [procesar_texto(t) for t in ["No me gustaron los productos, aunque la entrega fue rapida.", "El producto funciona, pero no cumple lo prometido."]]
    (OUT / "ejemplos_nlp.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = pd.read_csv(OUT / "predicciones_test_local.csv")
    predictions["indice_test"] = predictions.index
    predictions["posible_causa"] = predictions.texto.map(causa)
    manual = {
        4: "Lectura manual: describe rotura y frustracion; lenguaje negativo pese a etiqueta de tres estrellas. Desacuerdo entre puntuacion y texto.",
        8: "Lectura manual: aun no ha usado el comedero y refiere montaje complicado. Evidencia negativa sobre montaje, contexto insuficiente del uso.",
        15: "Lectura manual: producto en perfecto estado y caja deteriorada. Valoracion mixta explicita con pero; el modelo prioriza la parte positiva.",
        22: "Lectura manual: valora bien el producto pero informa manchas y dosificador roto. Valoracion mixta explicita.",
        27: "Lectura manual: defectos del producto junto con valoracion positiva del embalaje. La etiqueta tres estrellas agrega aspectos distintos."
    }
    for indice, explicacion in manual.items():
        predictions.loc[predictions.indice_test == indice, "posible_causa"] = explicacion
    predictions.to_csv(OUT / "analisis_ejemplos_local.csv", index=False, encoding="utf-8-sig")
    seleccion = pd.concat([predictions[predictions.indice_test.isin([15, 22, 27])], pd.concat([predictions[(predictions.real == label) & (predictions.correcto == correct)].head(2)
        for label in ["neutro", "negativo", "positivo"] for correct in [False, True]])]).drop_duplicates("indice_test")
    public = seleccion.drop(columns=["texto"])
    public.to_csv(OUT / "ejemplos_sin_texto.csv", index=False, encoding="utf-8-sig")
    comparacion = pd.read_csv(OUT / "comparacion_modelos.csv")
    nlp = comparacion[comparacion.modelo != "Mayoritaria"].sort_values("f1_macro_cv", ascending=False).drop_duplicates(["modelo", "preprocesamiento", "ngramas"])
    nlp.to_csv(OUT / "comparacion_nlp.csv", index=False, encoding="utf-8-sig")
    metricas_path = ROOT / OUT.name.replace("resultados_", "artifacts_", 1) / "metricas.json"
    (OUT / "metricas.json").write_text(metricas_path.read_text(encoding="utf-8"), encoding="utf-8")
    generar_notebook()
    generar_informe(comparacion, predictions, meta)
    print("Evidencia, notebook ejecutado y metadatos guardados.")

def generar_informe(comparacion, predictions, meta):
    metrics = json.loads((OUT / "metricas.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    best = comparacion.sort_values("f1_macro_cv", ascending=False).drop_duplicates("modelo")
    cols = ["modelo", "preprocesamiento", "ngramas", "hiperparametro", "f1_macro_cv", "desviacion_cv"]
    # Fixed C isolates NLP and ngram effects instead of changing multiple factors.
    controlled = comparacion[(comparacion.modelo.str.contains("log", case=False)) & (comparacion.hiperparametro == 0.5)]
    doc = """# Resultados experimentales de OpinIA - Kelly Mendoza

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

"""
    doc += tabla(best[cols].round(5))
    doc += "\n\n### Comparacion controlada de NLP y ngramas\n\nRegresion logistica con C=0.5 fijo; solo cambia NLP o rango de ngramas.\n\n" + tabla(controlled[cols].round(5))
    doc += "\n\n### Tabla completa de configuraciones\n\n" + tabla(comparacion[cols].round(5))
    doc += """

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

"""
    doc += tabla(pd.read_csv(OUT / "ejemplos_sin_texto.csv").drop(columns=["posible_causa"]).round(5))
    doc += "\n\n## Ejemplos antes/despues de NLP\n\n"
    for ex in json.loads((OUT / "ejemplos_nlp.json").read_text(encoding="utf-8")):
        doc += f"- Original artificial: {ex['original']}\n- Stemming: {' '.join(ex['stemming'])}\n- Lematizacion: {' '.join(ex['lematizacion'])}\n\n"
    doc += fr"""
## Reproducibilidad y proteccion de artefactos

Desde PowerShell, en C:/Proyecto_Topicos, con dependencias instaladas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reproducibles-kelly.txt
.\.venv\Scripts\python.exe -m src.models.reproducir_kelly --revision {manifest['revision_dataset']} --run-id {OUT.name.replace("resultados_", "", 1)}
```

El comando se ejecuto realmente para esta evidencia. Para otra ejecucion cambia run-id, por ejemplo `finales_kelly_revision2`, para conservar los anteriores. El prefijo `finales_kelly` mantiene excluidos los modelos y textos privados. El orquestador reutiliza preparar_datos y entrenar; conserva JSONL crudos en data/raw/<revision>, genera muestra separada, exige igualdad con CSV historicos, ejecuta CV, evalua el ganador, serializa, verifica predict/proba tras recarga y ejecuta el notebook. Descarga inicial requiere Internet; posteriores lecturas usan los crudos conservados. Los hashes permiten auditar integridad; no se garantiza que el proveedor mantenga el espejo disponible.

Python observado: {manifest['python']}. Versiones exactas de todas las dependencias, spaCy y modelo es_core_news_sm 3.8.0 quedan fijadas en requirements-reproducibles-kelly.txt. No se promete compatibilidad de joblib con otras versiones. El backend carga el pipeline; necesita el modulo compartido importable y el recurso spaCy instalado, sin reconstruir reglas ni reentrenar.

Hashes de produccion antes/despues identicos: {manifest['sha256_produccion_antes'] == manifest['sha256_produccion_despues']}. Los artefactos finales son copias aisladas, y no reemplazan artifacts/pipeline.joblib.

Resultados de esta ejecucion: `{OUT.name}/`. Metricas agregadas y configuraciones son publicables; las rese?as y errores con texto no.

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
git add -- {OUT.name}/comparacion_modelos.csv {OUT.name}/comparacion_nlp.csv {OUT.name}/calidad_datos.json {OUT.name}/ejemplos_nlp.json {OUT.name}/ejemplos_sin_texto.csv {OUT.name}/metadata_reproduccion.json {OUT.name}/manifest.json {OUT.name}/metricas.json
git diff --cached --name-only
git commit -m "feat(ml): registrar comparaciones de NLP y modelos"
git add -- tests/test_experimentos_finales_kelly.py {OUT.name}/pruebas.txt
git diff --cached --name-only
git commit -m "test(ml): verificar reproducibilidad y predicciones"
git push -u origin feature/modelos-kelly
```

Estos comandos no incluyen README modificado previamente, carpetas de revisiones anteriores ni .bak. Si se migra a feature/experimentos-finales-kelly desde main actualizado con arbol limpio, el push debe usar ese nombre. No mover las modificaciones actuales sin preservar y revisar los commits.

## Pull Request sugerido

Titulo: `feat(ml): completar experimentos reproducibles y evidencia de Kelly`

Descripcion: Completa la trazabilidad del dataset con revision y hashes, ejecuta EDA y registra 16 configuraciones NLP/modelo y baseline con los mismos folds. Reproduce la regresion logistica existente (F1 CV 0.59622; F1 test 0.60431), conserva el artefacto funcional y agrega analisis de errores y un comando desde crudos. Incluye pruebas de reproducibilidad, serializacion y carga desde backend. Los textos del corpus quedan locales por sus condiciones de uso. Revisar la distribucion previa de CSV y el endpoint de metricas durante la integracion.

En GitHub: Compare & pull request, base main, compare la rama publicada; pegar esta descripcion y solicitar revision a Isaias mediante su cuenta `IsaiasGutierrezTeran`. Revisar Files changed, confirmar que no hay corpus ni archivos locales, y completar la revision antes de merge. No se abrio ni envio una solicitud automaticamente.
"""
    (ROOT / "docs/resultados_experimentales.md").write_text(doc, encoding="utf-8")

if __name__ == "__main__":
    generar()
