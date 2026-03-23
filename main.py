from pathlib import Path
from pipeline import config, extract_text
import logging
import subprocess
import time
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# fichier de sortie attendu par étape
STEP_OUTPUTS = {
    "sentences"           : config.SENTENCES_PATH,
    "embeddings"          : config.EMBEDDINGS_PATH,
    "features_engineering": config.FEATURES_PATH,
    "merger"              : config.FINAL_PATH,
    "label"               : config.FINAL_LABELED_PATH,
    "modelise"            : config.MODELS_DIR / "evaluation.json",
}


def run_pipeline(steps=None, force=False):
    all_steps = {
        "index":   lambda: extract_text.index_database(config.Path_Sciencespo),
        "open":    lambda: extract_text.open_database(config.Path_Sciencespo),
        "extract": lambda: extract_text.extract_to_txt(),
        "sentences": lambda: subprocess.run([
                        "python", "-m", "pipeline.sentences",
                        "--data_dir",      "data/text_files/",
                        "--output",        str(config.SENTENCES_PATH),
                        "--max_sentences", "110000",
                    ], check=True),
        "embeddings": lambda: subprocess.run([
                        "python", "-m", "pipeline.embedder",
                        "--input",      str(config.SENTENCES_PATH),
                        "--output",     str(config.EMBEDDINGS_PATH),
                        "--model",      config.BERT_MODEL,
                        "--batch_size", str(config.BATCH_SIZE),
                    ], check=True),
        "features_engineering": lambda: subprocess.run([
                        "python", "-m", "pipeline.features_engineering",
                        "--input",  str(config.SENTENCES_PATH),
                        "--output", str(config.FEATURES_PATH),
                    ], check=True),
        "merger": lambda: subprocess.run([
                        "python", "-m", "pipeline.merger",
                        "--embeddings", str(config.EMBEDDINGS_PATH),
                        "--features",   str(config.FEATURES_PATH),
                        "--output",     str(config.FINAL_PATH),
                    ], check=True),
        "label": lambda: subprocess.run([
                        "python", "-m", "pipeline.labeler",
                        "--labels_csv", str(config.LABELS_CSV),
                        "--input",      str(config.FINAL_PATH),
                        "--output",     str(config.FINAL_LABELED_PATH),
                    ], check=True),
        "modelise": lambda: subprocess.run([
                        "python", "-m", "pipeline.modelisation",
                        "--input",  str(config.FINAL_LABELED_PATH),
                        "--full",   str(config.FINAL_PATH),
                        "--output", str(config.MODELS_DIR) + "/",
                    ], check=True),
        "dashboard": lambda: subprocess.run([
                        "streamlit", "run", "dashboard.py",
                    ], check=True),
        "statistique_resume": lambda: subprocess.run([
                        "streamlit", "run", "statistique_resume.py",
                    ], check=True),
    }

    steps = steps or list(all_steps.keys())

    # validation des noms d'étapes
    unknown = [s for s in steps if s not in all_steps]
    if unknown:
        logging.error(f"Étapes inconnues : {unknown} - Disponibles : {list(all_steps)}")
        raise SystemExit(1)

    start_time0 = time.perf_counter()

    for step_name in steps:
        # checkpoint (skip si output existe déjà)
        output = STEP_OUTPUTS.get(step_name)
        if output and Path(output).exists() and not force:
            logging.info(f"Skip '{step_name}' — output déjà présent : {output}")
            continue

        start_time = time.perf_counter()
        logging.info(f"Running step: {step_name}")

        try:
            all_steps[step_name]()
        except subprocess.CalledProcessError as e:
            logging.error(f"Étape '{step_name}' échouée (exit code {e.returncode})")
            raise SystemExit(1)

        elapsed = time.perf_counter() - start_time
        print(f"Execution time {step_name}: {elapsed:.4f} seconds")

    elapsed_total = time.perf_counter() - start_time0
    print("")
    print(f"Execution time (pipeline): {elapsed_total:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", nargs="+", default=None,
        help="Étapes à exécuter (ex: --steps sentences embeddings features_engineering merger label modelise)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forcer la réexécution même si l'output existe déjà",
    )
    args = parser.parse_args()
    run_pipeline(args.steps, force=args.force)
