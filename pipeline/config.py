from pathlib import Path
from .utils import load_lexicon, load_lexicons

Path_Sciencespo = Path("data/sciencespo-archelec-20260217-121320.sqlite")

############### TXT TO DATASET - SENTENCE.PY ####################
ENCODING      = "utf-8"
NLTK_LANGUAGE = "french"

# ── Labels d'annotation valides ──────────────────────────────────
# La comparaison 2-classes / 3-classes se fait à l'entraînement :
#   df_2class = df[df["label"].isin(["langue_de_bois", "non_langue_de_bois"])]
#   df_3class = df[df["label"].notna()]
VALID_LABELS = {"langue_de_bois", "non_langue_de_bois", "autre"}

############### EMBEDDER MODEL               ####################
BERT_MODEL  = "camembert-base"
BATCH_SIZE  = 256
MAX_LENGTH  = 128

############### Features Engineering          ###################
SPACY_MODEL       = "fr_core_news_md"
SENTIMENT_MODEL   = "cmarkea/distilcamembert-base-sentiment"

############### LEXICONS              ##################
VAGUE_WORDS = load_lexicons(
    "data/lexicons/dictionnaire_final_clean.txt"
)

MODAL_VERBS = load_lexicon("data/lexicons/modal_verbs.txt")

# ── Chemins des outputs ───────────────────────────────────────────
OUTPUTS_DIR          = Path("outputs")
SENTENCES_PATH       = OUTPUTS_DIR / "sentences.parquet"
EMBEDDINGS_PATH      = OUTPUTS_DIR / "embeddings.parquet"
FEATURES_PATH        = OUTPUTS_DIR / "features.parquet"
FINAL_PATH           = OUTPUTS_DIR / "final.parquet"
FINAL_LABELED_PATH   = OUTPUTS_DIR / "final_labeled.parquet"
FINAL_PREDICTED_PATH = OUTPUTS_DIR / "final_predicted.parquet"
MODELS_DIR           = OUTPUTS_DIR / "models"
LABELS_CSV           = Path("data/labels/annotation_sample.csv")

# ── Schémas attendus ──────────────────────────────────────────────
SCHEMA_SENTENCES     = {"PRIMARY_KEY", "doc_id", "date", "classe", "sentence", "filter_ratio"}
SCHEMA_EMBEDDINGS    = {"PRIMARY_KEY", "embedding"}
SCHEMA_FINAL         = SCHEMA_SENTENCES | {"embedding"}
SCHEMA_FINAL_LABELED = SCHEMA_FINAL | {"label"}


def validate_schema(df, required_cols: set, name: str = "") -> None:
    """Lève ValueError si des colonnes attendues sont absentes du DataFrame."""
    missing = required_cols - set(df.columns)
    if missing:
        label = f"[{name}] " if name else ""
        raise ValueError(
            f"{label}Colonnes manquantes : {sorted(missing)}\n"
            f"Colonnes présentes : {sorted(df.columns)}"
        )
