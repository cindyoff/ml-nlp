from pipeline import config, embedder, extract_text, features_engineering, sentences, labeler
import logging
import subprocess

def run_pipeline(steps=None):
    all_steps = {
        "index":     lambda: extract_text.index_database(config.Path_Sciencespo),
        "open":      lambda: extract_text.open_database(config.Path_Sciencespo),
        "extract":   lambda: extract_text.extract_to_txt(),
        "sentences": lambda: subprocess.run([
                        "python", "-m", "pipeline.sentences",
                        "--data_dir", "outputs/text_files/",
                        "--output",   "outputs/sentences.parquet",
                    ], check=True),
        "embeding":     lambda: subprocess.run([
                        "python", "-m", "pipeline.embedder",
                        "--input",      "outputs/sentences.parquet",
                        "--output",     "outputs/embeddings.parquet",
                        "--model",      config.BERT_MODEL,
                        "--batch_size", str(config.BATCH_SIZE),
                    ], check=True),
        "features_engineering":     lambda: subprocess.run([
                        "python", "-m", "pipeline.features_engineering",
                        "--input",      "outputs/sentences.parquet",
                        "--output",     "outputs/features.parquet",
                    ], check=True), 
        "labeling":     lambda: subprocess.run([
                        "python", "-m", "pipeline.labeler",
                        "--input",  "outputs/sentences.parquet",
                        "--output", "outputs/scores.parquet",
], check=True)
    }

    steps = steps or all_steps.keys()

    for step_name in steps:
        logging.info(f"Running step : {step_name}")
        all_steps[step_name]()