"""
merger.py
=========
Joint embeddings.parquet + features.parquet sur PRIMARY_KEY
pour produire un seul final.parquet avec toutes les colonnes.

Usage :
    python -m pipeline.merger \
        --embeddings outputs/embeddings.parquet \
        --features   outputs/features.parquet \
        --output     outputs/final.parquet
"""

import argparse
from pathlib import Path
import pandas as pd
from .config import validate_schema


def merge(embeddings_path: str, features_path: str, output_path: str) -> None:
    print(f"\n📂 Chargement embeddings : {embeddings_path}")
    df_emb = pd.read_parquet(embeddings_path)
    print(f"   {len(df_emb)} lignes  |  colonnes : {list(df_emb.columns)}")

    print(f"📂 Chargement features  : {features_path}")
    df_feat = pd.read_parquet(features_path)
    print(f"   {len(df_feat)} lignes  |  colonnes : {list(df_feat.columns)}\n")

    validate_schema(df_emb,  {"PRIMARY_KEY", "embedding"}, "embeddings.parquet")
    validate_schema(df_feat, {"PRIMARY_KEY"},               "features.parquet")

    # Vérification de l'intégrité des clés
    emb_keys  = set(df_emb["PRIMARY_KEY"])
    feat_keys = set(df_feat["PRIMARY_KEY"])
    missing_in_feat = emb_keys - feat_keys
    missing_in_emb  = feat_keys - emb_keys
    if missing_in_feat:
        print(f"  ⚠  {len(missing_in_feat)} clés présentes dans embeddings mais absentes de features")
    if missing_in_emb:
        print(f"  ⚠  {len(missing_in_emb)} clés présentes dans features mais absentes d'embeddings")

    # Jointure gauche : features comme base, on y attache la colonne embedding
    # (doc_id, date, classe déjà présents dans features — pas de duplication)
    df_final = df_feat.merge(
        df_emb[["PRIMARY_KEY", "embedding"]],
        on="PRIMARY_KEY",
        how="left",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out, index=False)

    print(f"💾 final.parquet sauvegardé → {out}")
    print(f"   Shape : {df_final.shape}")
    print(f"   Colonnes : {list(df_final.columns)}\n")


def main():
    parser = argparse.ArgumentParser(description="Merge embeddings + features → final.parquet")
    parser.add_argument("--embeddings", default="outputs/embeddings.parquet", help="Parquet des embeddings")
    parser.add_argument("--features",   default="outputs/features.parquet",   help="Parquet des features")
    parser.add_argument("--output",     default="outputs/final.parquet",      help="Parquet de sortie")
    args = parser.parse_args()

    merge(args.embeddings, args.features, args.output)


if __name__ == "__main__":
    main()
