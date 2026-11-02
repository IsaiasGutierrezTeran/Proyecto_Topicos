"""Comparar TF-IDF y clasificadores con CV; evaluar una única vez el ganador.

    python -m src.models.entrenar
    python -m src.models.entrenar --nlp-completo  # requiere módulo NLP de Isaías
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.models.adaptador_nlp import AdaptadorNLP

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["negativo", "neutro", "positivo"]
SEED = 42


def leer_datos(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecuta antes: python -m src.models.preparar_datos"
        )
    df = pd.read_csv(ruta, encoding="utf-8-sig").dropna(subset=["texto", "sentimiento"])
    if len(df) < 9:
        raise ValueError("Necesitas al menos 9 ejemplos de entrenamiento/prueba.")
    desconocidas = set(df["sentimiento"]) - set(LABELS)
    if desconocidas:
        raise ValueError(f"Etiquetas inesperadas: {desconocidas}")
    return df


def experimentos(df_train: pd.DataFrame, metodos: list[str]):
    X = df_train["texto"].astype(str).tolist()
    y = df_train["sentimiento"].astype(str).tolist()
    if min(pd.Series(y).value_counts()) < 3:
        raise ValueError("Cada clase necesita mínimo 3 ejemplos para la validación cruzada.")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    catalogo = [
        ("Naive Bayes", MultinomialNB(), {"clasificador__alpha": [0.5, 1.0]}),
        ("Regresión logística", LogisticRegression(max_iter=1500, random_state=SEED),
         {"clasificador__C": [0.5, 1.5]}),
    ]
    base_scores = cross_val_score(DummyClassifier(strategy="most_frequent"),
                                  np.zeros((len(y), 1)), y, cv=cv, scoring="f1_macro")
    filas = [{"configuracion": "mayoritaria", "modelo": "Mayoritaria",
              "preprocesamiento": "ninguno", "ngramas": "no aplica",
              "hiperparametro": None, "f1_macro_cv": float(base_scores.mean()),
              "desviacion_cv": float(base_scores.std()),
              **{f"fold_{i+1}": float(v) for i, v in enumerate(base_scores)}}]
    ganador = None
    mejor_score = -1.0
    for nombre, clasificador, ajustes in catalogo:
        print(f"Evaluando {nombre} con CV de 3 folds...")
        pipeline = Pipeline([
            ("nlp", AdaptadorNLP()),
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                     max_features=5000, min_df=1, max_df=0.95, sublinear_tf=True)),
            ("clasificador", clasificador),
        ])
        param_grid = {
            "nlp__metodo": metodos,
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            **ajustes,
        }
        busqueda = GridSearchCV(
            pipeline, param_grid=param_grid, scoring="f1_macro", cv=cv,
            n_jobs=1, refit=True, error_score="raise"
        )
        busqueda.fit(X, y)
        for i, params in enumerate(busqueda.cv_results_["params"]):
            filas.append({
                "configuracion": f"{nombre}/{params['nlp__metodo']}/{params['tfidf__ngram_range']}/{params.get('clasificador__alpha', params.get('clasificador__C'))}",
                "parametros_tfidf": json.dumps({**pipeline.named_steps["tfidf"].get_params(),
                    "ngram_range": params["tfidf__ngram_range"]}, default=str, ensure_ascii=False),
                **{f"fold_{fold+1}": float(busqueda.cv_results_[f"split{fold}_test_score"][i]) for fold in range(3)},
                "modelo": nombre,
                "preprocesamiento": params["nlp__metodo"],
                "ngramas": "unigramas" if params["tfidf__ngram_range"] == (1, 1)
                          else "unigramas_y_bigramas",
                "hiperparametro": float(params.get("clasificador__alpha", params.get("clasificador__C", 0))),
                "f1_macro_cv": round(float(busqueda.cv_results_["mean_test_score"][i]), 5),
                "desviacion_cv": round(float(busqueda.cv_results_["std_test_score"][i]), 5),
            })
        if busqueda.best_score_ > mejor_score:
            mejor_score = float(busqueda.best_score_)
            ganador = (nombre, busqueda.best_estimator_, busqueda.best_params_)
    assert ganador is not None
    return ganador, filas, mejor_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nlp-completo", action="store_true",
                   help="Activa stemming y lematización definidos por Isaías")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    p.add_argument("--results-dir", type=Path, default=ROOT / "resultados")
    args = p.parse_args()
    if (args.artifacts_dir / "pipeline.joblib").exists():
        raise FileExistsError("El modelo de destino ya existe; utiliza otra carpeta --artifacts-dir")
    metodos = ["stemming", "lematizacion"] if args.nlp_completo else ["basico"]

    train = leer_datos(args.data_dir / "train.csv")
    test = leer_datos(args.data_dir / "test.csv")
    # Comprobar que una misma reseña no aparezca en ambos conjuntos.
    overlap = set(train["texto"].str.casefold()) & set(test["texto"].str.casefold())
    if overlap:
        raise ValueError(f"Fuga de datos: {len(overlap)} textos repetidos entre train/test")

    ganador, filas, cv_score = experimentos(train, metodos)
    nombre, modelo, parametros = ganador
    # Baseline mayoritaria: predictor trivial para comparar el modelo elegido.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[["texto"]], train["sentimiento"])
    y_base = baseline.predict(test[["texto"]])
    baseline_f1 = float(f1_score(test["sentimiento"], y_base, labels=LABELS, average="macro", zero_division=0))

    X_test = test["texto"].astype(str).tolist()
    y_test = test["sentimiento"].astype(str).tolist()
    prediccion = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)
    nombres_clase = list(modelo.classes_)
    confianza = [float(probabilidades[i, nombres_clase.index(prediccion[i])])
                 for i in range(len(prediccion))]

    metricas = {
        "tfidf": modelo.named_steps["tfidf"].get_params(),
        "vocabulario_efectivo": len(modelo.named_steps["tfidf"].vocabulary_),
        "modelo_ganador": nombre,
        "mejores_parametros": {
            k: list(v) if isinstance(v, tuple) else v for k, v in parametros.items()
        },
        "f1_macro_cv": round(cv_score, 5),
        "f1_macro_baseline_mayoritaria": round(baseline_f1, 5),
        "preprocesamiento_evaluado": metodos,
        "pipeline_completo_con_nlp_de_isaias": bool(args.nlp_completo),
        "accuracy_test": round(float(accuracy_score(y_test, prediccion)), 5),
        "precision_macro_test": round(float(precision_score(y_test, prediccion, labels=LABELS,
                                                              average="macro", zero_division=0)), 5),
        "recall_macro_test": round(float(recall_score(y_test, prediccion, labels=LABELS,
                                                      average="macro", zero_division=0)), 5),
        "f1_macro_test": round(float(f1_score(y_test, prediccion, labels=LABELS,
                                                  average="macro", zero_division=0)), 5),
        "clases": LABELS,
        "matriz_confusion": confusion_matrix(y_test, prediccion, labels=LABELS).tolist(),
        "reporte_clasificacion": classification_report(y_test, prediccion, labels=LABELS,
                                                        output_dict=True, zero_division=0),
        "train_ejemplos": int(len(train)), "test_ejemplos": int(len(test)),
        "semilla": SEED,
        "nota": "La selección se hizo con CV sobre train; test se usó solo para evaluar el ganador."
    }
    artefactos = args.artifacts_dir
    resultados = args.results_dir
    artefactos.mkdir(parents=True, exist_ok=True)
    resultados.mkdir(parents=True, exist_ok=True)
    if (artefactos / "pipeline.joblib").exists():
        raise FileExistsError("El modelo de destino ya existe; utiliza otra carpeta --artifacts-dir")
    joblib.dump(modelo, artefactos / "pipeline.joblib")
    recargado = joblib.load(artefactos / "pipeline.joblib")
    if not np.array_equal(prediccion, recargado.predict(X_test)):
        raise RuntimeError("La serializacion cambio las predicciones")
    np.testing.assert_allclose(probabilidades, recargado.predict_proba(X_test))
    (artefactos / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(filas).sort_values("f1_macro_cv", ascending=False).to_csv(
        resultados / "comparacion_modelos.csv", index=False, encoding="utf-8-sig"
    )
    errores = pd.DataFrame({"texto": X_test, "real": y_test, "predicho": prediccion,
                            "confianza_prediccion": np.round(confianza, 4)})
    for j, clase in enumerate(nombres_clase):
        errores[f"p_{clase}"] = probabilidades[:, j]
    errores["correcto"] = errores["real"] == errores["predicho"]
    errores.to_csv(resultados / "predicciones_test_local.csv", index=False, encoding="utf-8-sig")
    errores = errores[~errores["correcto"]].sort_values(
        "confianza_prediccion", ascending=False
    )
    errores.head(25).to_csv(resultados / "analisis_errores.csv", index=False,
                           encoding="utf-8-sig")
    print(f"Ganador: {nombre} | F1 macro CV: {cv_score:.3f} | "
          f"F1 macro test: {metricas['f1_macro_test']:.3f}")
    print(f"Listo: {artefactos}/pipeline.joblib, metricas.json y {resultados}/*.csv")
    if not args.nlp_completo:
        print("AVISO: falta ejecutar --nlp-completo cuando Isaías entregue el módulo NLP")


if __name__ == "__main__":
    main()
