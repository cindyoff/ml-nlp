"""
Génère un CSV d'annotation à partir de sentences.parquet

Sélection de documents complets, avec stratification jusqu'à atteindre la cible de phrases

Résultat CSV : 1 ligne = 1 phrase avec colonne "label" vide pour remplissage ultérieur
"""

import argparse
import pandas as pd


def sample_documents(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """
    Sélection proportionnelle de documents complets avec stratification
    """
    groups   = df.groupby(["date", "classe"])
    total    = len(df)
    sampled_docs = []

    for (date, classe), group in groups:
        # proportion de phrases du groupe dans le corpus total
        proportion = len(group) / total
        # nombre de phrases cibles pour ce groupe
        group_target = int(target * proportion)

        # docs dispo du groupe, aléatoire
        docs = group["doc_id"].unique().tolist()
        rng  = __import__("random")
        rng.seed(seed)
        rng.shuffle(docs)

        # ajout de docs complets
        n_phrases = 0
        for doc_id in docs:
            doc_sentences = group[group["doc_id"] == doc_id]
            if n_phrases + len(doc_sentences) > group_target * 1.15:  # tolérance 15%
                break
            sampled_docs.append(doc_sentences)
            n_phrases += len(doc_sentences)

        print(f"[{date}/{classe}] {n_phrases} phrases ({len(sampled_docs)} docs au total jusqu'ici)")

    return pd.concat(sampled_docs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Génère un CSV pour l'annotation manuelle")
    parser.add_argument("--input",  default="outputs/sentences.parquet",
                        help="Parquet source (outputs/sentences.parquet)")
    parser.add_argument("--output", default="data/labels/annotation_sample.csv",
                        help="CSV de sortie")
    parser.add_argument("--target", default=1100, type=int,
                        help="Nombre cible de phrases (documents conservés entiers)")
    parser.add_argument("--seed",   default=42, type=int,
                        help="Graine aléatoire pour la reproductibilité")
    args = parser.parse_args()

    print(f"\nChargement : {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   {len(df)} phrases, {df['doc_id'].nunique()} documents\n")

    print(f"Échantillonnage stratifié et cible : {args.target} phrases")
    sample = sample_documents(df, target=args.target, seed=args.seed)

    # Colonnes utiles pour l'annotation + colonne label vide
    out = sample[["PRIMARY_KEY", "doc_id", "date", "classe", "sentence"]].copy()
    out["label"] = ""

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"\n{len(out)} phrases issues de {out['doc_id'].nunique()} documents")
    print(f"CSV sauvegardé → {args.output}\n")
    print("Remplis la colonne 'label' (langue_de_bois / non_langue_de_bois),")
    print("puis lance : python main.py --steps label\n")

if __name__ == "__main__":
    main()