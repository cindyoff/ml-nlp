from pathlib import Path
from .utils import load_lexicon, load_lexicons

Path_Sciencespo = Path("data/sciencespo-archelec-20260217-121320.sqlite")

# txt file into dataset
ENCODING      = "utf-8"
NLTK_LANGUAGE = "french"

# label annotation
VALID_LABELS = {"langue_de_bois", "non_langue_de_bois", "autre"}

# embedder model
BERT_MODEL  = "camembert-base"
BATCH_SIZE  = 256
MAX_LENGTH  = 128

# feature engineering
SPACY_MODEL       = "fr_core_news_md"
SENTIMENT_MODEL   = "cmarkea/distilcamembert-base-sentiment"

# lexicon
VAGUE_WORDS = load_lexicons(
    "data/lexicons/dictionnaire_final_clean.txt"
)
MODAL_VERBS = load_lexicon("data/lexicons/modal_verbs.txt")

# chemin output
OUTPUTS_DIR          = Path("outputs")
SENTENCES_PATH       = OUTPUTS_DIR / "sentences.parquet"
EMBEDDINGS_PATH      = OUTPUTS_DIR / "embeddings.parquet"
FEATURES_PATH        = OUTPUTS_DIR / "features.parquet"
FINAL_PATH           = OUTPUTS_DIR / "final.parquet"
FINAL_LABELED_PATH   = OUTPUTS_DIR / "final_labeled.parquet"
FINAL_PREDICTED_PATH = OUTPUTS_DIR / "final_predicted.parquet"
MODELS_DIR           = OUTPUTS_DIR / "models"
LABELS_CSV           = Path("data/labels/annotation_sample.csv")

# schemas attendus
SCHEMA_SENTENCES     = {"PRIMARY_KEY", "doc_id", "date", "classe", "sentence", "filter_ratio"}
SCHEMA_EMBEDDINGS    = {"PRIMARY_KEY", "embedding"}
SCHEMA_FINAL         = SCHEMA_SENTENCES | {"embedding"}
SCHEMA_FINAL_LABELED = SCHEMA_FINAL | {"label"}

# test
def validate_schema(df, required_cols: set, name: str = "") -> None:
    """ValueError si des colonnes attendues sont absentes du DataFrame"""
    missing = required_cols - set(df.columns)
    if missing:
        label = f"[{name}] " if name else ""
        raise ValueError(
            f"{label}Colonnes manquantes : {sorted(missing)}\n"
            f"Colonnes présentes : {sorted(df.columns)}"
        )
