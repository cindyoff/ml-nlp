"""
embedder.py
===========
Charge le parquet de phrases, encode chaque phrase avec CamemBERT
et sauvegarde un nouveau parquet où :
  - chaque ligne est une phrase
  - la colonne 'embedding' contient le vecteur float32 de 768 dimensions
  - les colonnes PRIMARY_KEY, doc_id, date, classe relient chaque phrase à sa source

Usage :
    python -m pipeline.embedder --input outputs/sentences.parquet --output outputs/embeddings.parquet
"""

import argparse
from pathlib import Path
from .config import BERT_MODEL, BATCH_SIZE, MAX_LENGTH, validate_schema
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel

# ──────────────────────────────────────────────
# CLASSE ENCODEUR
# ──────────────────────────────────────────────

class BertEmbedder:
    def __init__(self, model_name: str = BERT_MODEL):
        self.device = self._get_device()
        print(f"🖥  Appareil utilisé : {self.device}")

        print(f"🔄 Chargement du modèle '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).half().to(self.device)  # fp16
        self.model.eval()
        print("✅ Modèle chargé.\n")

    def encode(self, sentences: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            # Mean pooling sur les tokens non-padding (meilleur que CLS pour sentence-level)
            token_embeddings = output.last_hidden_state.float()
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            mean_embeddings = (
                (token_embeddings * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp(min=1e-9)
            ).cpu().numpy()

            all_embeddings.append(mean_embeddings)

            print(f"  Encodage : {min(i + batch_size, len(sentences))}/{len(sentences)} phrases", end="\r")

        print()
        return np.vstack(all_embeddings)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")         # Apple Silicon
        else:
            return torch.device("cpu")


# ──────────────────────────────────────────────
# CONSTRUCTION DU DATAFRAME FINAL
# ──────────────────────────────────────────────

def build_embedding_dataframe(df_sentences: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Construit un DataFrame avec :
      - les métadonnées (PRIMARY_KEY, doc_id, date, classe)
      - une colonne 'embedding' contenant le vecteur float32 de 768 dimensions
    """
    meta_cols = ["PRIMARY_KEY", "doc_id", "date", "classe"]
    df_meta = df_sentences[meta_cols].reset_index(drop=True).copy()
    df_meta["embedding"] = list(embeddings.astype(np.float32))
    return df_meta


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embedding BERT → parquet colonne embedding[768]")
    parser.add_argument("--input",      required=True,                        help="Parquet d'entrée (ex: outputs/sentences.parquet)")
    parser.add_argument("--output",     default="outputs/embeddings.parquet", help="Parquet de sortie")
    parser.add_argument("--model",      default=BERT_MODEL,                   help="Modèle BERT de HuggingFace")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int,         help="Taille des batchs")
    args = parser.parse_args()

    # ── Chargement ──
    print(f"\n📂 Chargement de : {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   {len(df)} phrases chargées\n")
    validate_schema(df, {"PRIMARY_KEY", "doc_id", "date", "classe", "sentence"}, "sentences.parquet")

    # ── Encodage ──
    encoder    = BertEmbedder(model_name=args.model)
    sentences  = df["sentence"].tolist()
    embeddings = encoder.encode(sentences, batch_size=args.batch_size)

    # ── Construction du DataFrame final ──
    df_final = build_embedding_dataframe(df, embeddings)

    # ── Sauvegarde avec pyarrow (schéma explicite pour la colonne list) ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("PRIMARY_KEY", pa.string()),
        ("doc_id",      pa.string()),
        ("date",        pa.string()),
        ("classe",      pa.string()),
        ("embedding",   pa.list_(pa.float32())),
    ])
    table = pa.Table.from_pydict(
        {
            "PRIMARY_KEY": df_final["PRIMARY_KEY"].tolist(),
            "doc_id":      df_final["doc_id"].tolist(),
            "date":        df_final["date"].tolist(),
            "classe":      df_final["classe"].tolist(),
            "embedding":   df_final["embedding"].tolist(),
        },
        schema=schema,
    )
    pq.write_table(table, output_path, compression="snappy")

    print(f"\n💾 Parquet sauvegardé → {output_path}")
    print(f"   {len(df_final)} phrases × 1 colonne embedding (768 dims, float32, snappy)")
    print(f"   Colonnes : {list(df_final.columns)}\n")


if __name__ == "__main__":
    main()
