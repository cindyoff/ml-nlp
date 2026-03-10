"""
features_engineering.py
========================
Charge le parquet de phrases et ajoute des features linguistiques :
  1. Concreteness score  (chiffres, dates, montants, % numériques)
  2. Vague words         (lexique de mots vagues)
  3. Modal verbs         (verbes modaux français)
  4. Named entities      (organisations, lieux, lois via spaCy)
  5. Sentiment intensity (via transformers camembert-sentiment)

Usage :
    python -m pipeline.features_engineering \
        --input  outputs/sentences.parquet \
        --output outputs/features.parquet
"""

import re
import argparse
from pathlib import Path
from config import SPACY_MODEL, SENTIMENT_MODEL
import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline

# ── Lexique mots vagues (français) ──
VAGUE_WORDS = {
    "ensemble", "avenir", "futur", "progrès", "valeurs", "force", "espoir",
    "engagement", "ambition", "volonté", "dynamique", "défi", "enjeu",
    "cohésion", "solidarité", "confiance", "dialogue", "concertation",
    "mobilisation", "responsabilité", "excellence", "innovation", "vision",
    "transparence", "efficacité", "modernisation", "transformation",
    "développement", "croissance", "équilibre", "harmonie", "bien-être",
}

# ── Verbes modaux français ──
MODAL_VERBS = {
    "devoir", "pouvoir", "vouloir", "falloir", "savoir",
    "doi", "doit", "doivent", "devons", "devez", "devrait", "devraient",
    "peut", "peuvent", "pouvons", "pourrait", "pourraient",
    "faut", "faudrait",
    "veux", "veut", "voulons", "voudrais", "voudrait",
}

# ── Patterns regex ──
PATTERN_DIGIT       = re.compile(r'\d+')
PATTERN_DATE        = re.compile(
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'           # 01/01/2020
    r'|\b(janvier|février|mars|avril|mai|juin|juillet|août'
    r'|septembre|octobre|novembre|décembre)\s+\d{4}\b'       # janvier 2020
    r'|\b\d{4}\b',                                           # 2020
    re.IGNORECASE
)
PATTERN_MONEY       = re.compile(
    r'\b\d+[\s]?(?:€|euros?|milliards?|millions?|milliard|million)\b',
    re.IGNORECASE
)
PATTERN_PERCENT     = re.compile(r'\d+[\.,]?\d*\s*%')


# ──────────────────────────────────────────────
# 1. CONCRETENESS SCORE
# ──────────────────────────────────────────────

def compute_concreteness(sentence: str) -> dict:
    words       = sentence.split()
    n_words     = max(len(words), 1)
    digits      = PATTERN_DIGIT.findall(sentence)
    dates       = PATTERN_DATE.findall(sentence)
    money       = PATTERN_MONEY.findall(sentence)
    percents    = PATTERN_PERCENT.findall(sentence)

    return {
        "n_digits"         : len(digits),
        "n_dates"          : len([d for d in dates if any(d)]),
        "n_money"          : len(money),
        "n_percent"        : len(percents),
        "numeric_ratio"    : len(digits) / n_words,
    }


# ──────────────────────────────────────────────
# 2. VAGUE WORDS
# ──────────────────────────────────────────────

def compute_vague_words(sentence: str) -> dict:
    words       = sentence.lower().split()
    n_words     = max(len(words), 1)
    vague_found = [w for w in words if w in VAGUE_WORDS]

    return {
        "n_vague_words"    : len(vague_found),
        "vague_ratio"      : len(vague_found) / n_words,
    }


# ──────────────────────────────────────────────
# 3. MODAL VERBS
# ──────────────────────────────────────────────

def compute_modal_verbs(sentence: str) -> dict:
    words        = sentence.lower().split()
    n_words      = max(len(words), 1)
    modals_found = [w for w in words if w in MODAL_VERBS]

    return {
        "n_modal_verbs"    : len(modals_found),
        "modal_ratio"      : len(modals_found) / n_words,
    }


# ──────────────────────────────────────────────
# 4. NAMED ENTITIES (spaCy)
# ──────────────────────────────────────────────

def compute_named_entities(doc) -> dict:
    """Reçoit un doc spaCy déjà parsé."""
    n_org   = sum(1 for ent in doc.ents if ent.label_ == "ORG")
    n_loc   = sum(1 for ent in doc.ents if ent.label_ in ("LOC", "GPE"))
    n_law   = sum(1 for ent in doc.ents if ent.label_ == "LAW")
    n_total = len(doc.ents)

    return {
        "n_ent_org"        : n_org,
        "n_ent_loc"        : n_loc,
        "n_ent_law"        : n_law,
        "n_ent_total"      : n_total,
    }


# ──────────────────────────────────────────────
# 5. SENTIMENT INTENSITY
# ──────────────────────────────────────────────

class SentimentScorer:
    def __init__(self, model_name: str = SENTIMENT_MODEL):
        print(f"🔄 Chargement du modèle sentiment '{model_name}'...")
        self.pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,           # retourne tous les scores
            truncation=True,
            max_length=128,
        )
        print("✅ Modèle sentiment chargé.\n")

    def score_batch(self, sentences: list[str]) -> list[dict]:
        results = self.pipe(sentences)
        output  = []
        for res in results:
            scores = {r["label"].lower(): r["score"] for r in res}
            # Intensité = distance à la neutralité
            positive   = scores.get("positive", scores.get("5 stars", 0.0))
            negative   = scores.get("negative", scores.get("1 star",  0.0))
            intensity  = abs(positive - negative)
            output.append({
                "sentiment_positive"   : positive,
                "sentiment_negative"   : negative,
                "sentiment_intensity"  : intensity,
            })
        return output


# ──────────────────────────────────────────────
# PIPELINE COMPLET
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    sentences = df["sentence"].tolist()

    # ── spaCy ──
    print("🔄 Chargement de spaCy...")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        raise OSError(
            f"Modèle spaCy '{SPACY_MODEL}' introuvable.\n"
            f"Installez-le avec : python -m spacy download {SPACY_MODEL}"
        )
    print("✅ spaCy chargé.\n")

    # ── Sentiment ──
    scorer = SentimentScorer()

    # ── Calcul des features ──
    all_features = []
    print("⚙️  Calcul des features...\n")

    # NER en batch avec spaCy (plus rapide)
    docs = list(nlp.pipe(sentences, batch_size=64))

    # Sentiment en batch
    SENT_BATCH = 64
    sentiment_scores = []
    for i in range(0, len(sentences), SENT_BATCH):
        batch = sentences[i : i + SENT_BATCH]
        sentiment_scores.extend(scorer.score_batch(batch))
        print(f"  Sentiment : {min(i + SENT_BATCH, len(sentences))}/{len(sentences)}", end="\r")
    print()

    for i, (sentence, doc) in enumerate(zip(sentences, docs)):
        features = {}
        features.update(compute_concreteness(sentence))
        features.update(compute_vague_words(sentence))
        features.update(compute_modal_verbs(sentence))
        features.update(compute_named_entities(doc))
        features.update(sentiment_scores[i])
        all_features.append(features)

    df_features = pd.DataFrame(all_features)

    # ── Concreteness score global (combinaison des sous-features) ──
    df_features["concreteness_score"] = (
        df_features["n_digits"]
        + df_features["n_dates"] * 2
        + df_features["n_money"] * 3
        + df_features["n_percent"] * 2
        + df_features["n_ent_org"]
        + df_features["n_ent_loc"]
        + df_features["n_ent_law"] * 2
    )

    return pd.concat([df.reset_index(drop=True), df_features], axis=1)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature engineering pour la détection de langue de bois")
    parser.add_argument("--input",  required=True,                      help="Parquet d'entrée (ex: outputs/sentences.parquet)")
    parser.add_argument("--output", default="outputs/features.parquet", help="Parquet de sortie")
    args = parser.parse_args()

    print(f"\n📂 Chargement de : {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   {len(df)} phrases chargées\n")

    df_final = build_features(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    feature_cols = [c for c in df_final.columns if c not in df.columns]
    print(f"\n💾 Parquet sauvegardé → {output_path}")
    print(f"   Shape : {df_final.shape}")
    print(f"   {len(feature_cols)} nouvelles features : {feature_cols}\n")


if __name__ == "__main__":
    main()