from pathlib import Path
from utils import load_lexicons

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
VAGUE_WORDS = load_lexicons(
    "data/lexicons/vague_words.txt",
    "data/lexicons/langue_de_bois.txt"
)

