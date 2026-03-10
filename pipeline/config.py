from pathlib import Path
from pipeline.utils import load_lexicons

Path_Sciencespo = Path("data/sciencespo-archelec-20260217-121320.sqlite")

############### TXT TO DATASET - SENTENCE.PY ####################
ENCODING      = "utf-8"
NLTK_LANGUAGE = "french"

############### EMBEDDER MODEL               ####################
BERT_MODEL  = "camembert-base"
BATCH_SIZE  = 32
MAX_LENGTH  = 128

############### Features Engeneerirng         ###################
SPACY_MODEL       = "fr_core_news_md"
SENTIMENT_MODEL   = "cmarkea/distilcamembert-base-sentiment"

############### LEXICONS              ##################
LANGUE_DE_BOIS = load_lexicons(
    "dictionnaire/dictionnaire_final_clean.txt"
)