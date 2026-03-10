import pandas as pd
import spacy
from pipeline.config import LANGUE_DE_BOIS

def score_document(texte: str, dico: set, nlp) -> dict:
    """Score de langue de bois pour un document entier."""
    doc = nlp(texte.lower())
    texte_lemma = " ".join([token.lemma_ for token in doc])
    
    mots_total  = len(texte_lemma.split())
    mots_trouves = [mot for mot in dico if mot in texte_lemma]
    
    return {
        "n_langue_de_bois" : len(mots_trouves),
        "score_ldb"        : round(len(mots_trouves) / mots_total * 100, 4)
                             if mots_total > 0 else 0,
        "mots_detectes"    : mots_trouves,
    }

def label_corpus(
    df_sentences: pd.DataFrame,
    metadata_path: str = "data/archelect_search.csv"
) -> pd.DataFrame:
    """
    Calcule le score de langue de bois par document
    et joint avec les métadonnées Archelec.
    """
    nlp = spacy.load("fr_core_news_md")

    # Score par document (agrégation des phrases)
    textes_par_doc = (
        df_sentences.groupby("doc_id")["sentence"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    scores = textes_par_doc.apply(
        lambda row: pd.Series(score_document(row["sentence"], LANGUE_DE_BOIS, nlp)),
        axis=1
    )
    df_scores = pd.concat([textes_par_doc[["doc_id"]], scores], axis=1)

    # Jointure avec les métadonnées
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.rename(columns={"id": "doc_id"})

    df_final = df_scores.merge(
        metadata[[
            "doc_id",
            "date",
            "contexte-election",
            "titulaire-nom",
            "titulaire-prenom",
            "titulaire-profession",
            "titulaire-soutien",
        ]],
        on="doc_id",
        how="left"
    )

    return df_final