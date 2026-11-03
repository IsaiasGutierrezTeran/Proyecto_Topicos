"""Orquestador: reutiliza preparaci?n y entrenamiento, sin sustituir producci?n."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="c379a6705fec24a2493fa68e011692605f44e119", help="Commit versionado del espejo HF")
    parser.add_argument("--run-id", default="finales_kelly", help="Nombre de salida; usa otro nombre para repetir sin sobrescribir")
    args = parser.parse_args()
    if not re.fullmatch(r"finales_kelly[A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id debe empezar por finales_kelly y contener solo letras, numeros, guiones o subrayados")
    out = ROOT / f"resultados_{args.run_id}"
    artifacts = ROOT / f"artifacts_{args.run_id}"
    if (artifacts / "pipeline.joblib").exists():
        raise FileExistsError("Ya hay un experimento final. Conserva esa carpeta antes de repetir.")
    before = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "artifacts/pipeline.joblib", ROOT / "artifacts/metricas.json"]}
    revision = args.revision
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        parser.error("revision debe ser un SHA completo de 40 caracteres")
    out.mkdir(exist_ok=True)
    data = ROOT / "data/reproducido_kelly"
    def run(module, *options):
        subprocess.run([sys.executable, "-m", module, *map(str, options)], cwd=ROOT, check=True)
    run("src.models.preparar_datos", "--raw-dir", ROOT / "data/raw", "--revision", revision,
        "--output-dir", data, "--metadata", out / "metadata_reproduccion.json")
    import pandas as pd
    for split in ["train", "test"]:
        pd.testing.assert_frame_equal(pd.read_csv(data / f"{split}.csv"),
                                      pd.read_csv(ROOT / "data/processed" / f"{split}.csv"))
    run("src.models.entrenar", "--nlp-completo", "--data-dir", data,
        "--artifacts-dir", artifacts, "--results-dir", out)
    versions = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    manifest = {"revision_dataset": revision, "python": platform.python_version(),
                "dependencias": versions, "sha256_produccion_antes": before,
                "sha256_produccion_despues": {p: sha(ROOT / p) for p in before},
                "muestra_reproducida_identica": True,
                "sha256_fuentes": {str(p.relative_to(ROOT)): sha(p) for p in
                    sorted((ROOT / "src").rglob("*.py"))},
                "sha256_pipeline_nuevo": sha(artifacts / "pipeline.joblib")}
    assert manifest["sha256_produccion_antes"] == manifest["sha256_produccion_despues"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "requirements-reproducibles-kelly.txt").write_text(
        "# Entorno completo observado; Python " + platform.python_version() + "\n" +
        "\n".join(f"{k}=={v}" for k, v in sorted(versions.items()) if k.lower().replace("_", "-") != "es-core-news-sm") +
        "\nes-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl\n", encoding="utf-8")
    from src.models.evidencia_kelly import generar
    generar(out)
    print("Reproduccion completa. Modelo de produccion conservado; evidencia en", out)

if __name__ == "__main__":
    main()
