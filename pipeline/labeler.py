"""
labeler.py
==========
Lit un CSV d'annotation (PRIMARY_KEY, label) généré par sample_annotation.py
et ajoute la colonne "label" au parquet final.

Le CSV est la source de vérité des annotations manuelles :
  - PRIMARY_KEY : identifiant unique de la phrase
  - label       : valeur renseignée manuellement

Labels valides (définis dans config.VALID_LABELS) :
  langue_de_bois | non_langue_de_bois | autre

Deux fichiers sont produits automatiquement à partir de --output :
  - {output}               → toutes les phrases étiquetées (3 classes)
  - {stem}_2class{suffix}  → uniquement langue_de_bois + non_langue_de_bois

Usage :
    python -m pipeline.labeler \
        --labels_csv data/labels/annotation_sample.csv \
        --input      outputs/final.parquet \
        --output     outputs/final_labeled.parquet
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from .config import VALID_LABELS, validate_schema

logger = logging.getLogger(__name__)

_2CLASS_LABELS = {"langue_de_bois", "non_langue_de_bois"}


def apply_labels(labels_csv: str, parquet_path: str, output_path: str) -> None:
    csv_path = Path(labels_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {labels_csv}")

    print(f"\n📂 Chargement du parquet : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   {len(df)} phrases chargées")
    validate_schema(df, {"PRIMARY_KEY"}, "final.parquet")

    print(f"🏷  Chargement des labels : {labels_csv}  (valides : {VALID_LABELS})")
    df_labels = pd.read_csv(csv_path, encoding="utf-8", dtype=str)

    if "PRIMARY_KEY" not in df_labels.columns or "label" not in df_labels.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'PRIMARY_KEY' et 'label'.")

    # Garder uniquement les lignes avec un label renseigné
    df_labels = df_labels[df_labels["label"].notna() & (df_labels["label"].str.strip() != "")]
    df_labels = df_labels[["PRIMARY_KEY", "label"]].drop_duplicates("PRIMARY_KEY")

    # Validation des valeurs de labels
    invalid = set(df_labels["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Labels inconnus : {invalid}\n"
            f"Valeurs attendues : {VALID_LABELS}"
        )

    print(f"   {len(df_labels)} labels renseignés")
    for lbl, cnt in sorted(df_labels["label"].value_counts().items()):
        print(f"     {lbl} : {cnt}")

    # Jointure sur PRIMARY_KEY
    df = df.merge(df_labels, on="PRIMARY_KEY", how="left")

    labeled   = df["label"].notna().sum()
    unlabeled = df["label"].isna().sum()
    print(f"   Étiquetées     : {labeled}")
    print(f"   Non étiquetées : {unlabeled}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Sortie 1 : toutes les phrases (3 classes : langue_de_bois, non_langue_de_bois, autre)
    df.to_parquet(out, index=False)
    print(f"\n💾 3 classes → {out}  (shape : {df.shape})")

    # ── Sortie 2 : seulement langue_de_bois + non_langue_de_bois (2 classes)
    df_2class = df[df["label"].isin(_2CLASS_LABELS) | df["label"].isna()]
    out_2class = out.with_name(out.stem + "_2class" + out.suffix)
    df_2class.to_parquet(out_2class, index=False)
    print(f"💾 2 classes → {out_2class}  (shape : {df_2class.shape})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Applique les labels du CSV à final.parquet")
    parser.add_argument("--labels_csv", required=True,
                        help="CSV d'annotation (ex: data/labels/annotation_sample.csv)")
    parser.add_argument("--input",  required=True,
                        help="Parquet d'entrée (ex: outputs/final.parquet)")
    parser.add_argument("--output", default="outputs/final_labeled.parquet",
                        help="Parquet de sortie (base — deux fichiers générés automatiquement)")
    args = parser.parse_args()

    apply_labels(args.labels_csv, args.input, args.output)


if __name__ == "__main__":
    main()
