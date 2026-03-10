"""
embedder.py
===========
Charge le parquet de phrases, encode chaque phrase avec CamemBERT
et sauvegarde un nouveau parquet où :
  - chaque ligne est une phrase
  - les colonnes bert_0 ... bert_767 sont les 768 dimensions de l'embedding
  - la colonne doc_id relie chaque phrase à son document source

Usage :
    python -m pipeline.embedder --input outputs/sentences.parquet --output outputs/embeddings.parquet
"""

import argparse
from pathlib import Path
from pipeline.config import BERT_MODEL, BATCH_SIZE, MAX_LENGTH 
import numpy as np
import pandas as pd
import torch
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
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅ Modèle chargé.\n")

    def encode(self, sentences: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Encode une liste de phrases → array (n_phrases, 768) via token [CLS].
        """
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

            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            print(f"  Encodage : {min(i + batch_size, len(sentences))}/{len(sentences)} phrases", end="\r")

        print()
        return np.vstack(all_embeddings)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")        # GPU NVIDIA
        elif torch.backends.mps.is_available():
            return torch.device("mps")         # Apple Silicon ✅
        else:
            return torch.device("cpu")

    

# ──────────────────────────────────────────────
# CONSTRUCTION DU DATAFRAME FINAL
# ──────────────────────────────────────────────

def build_embedding_dataframe(df_sentences: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Construit un DataFrame avec :
      - les métadonnées (doc_id, date, classe, sentence, label)
      - une colonne par dimension BERT (bert_0 ... bert_767)
    """
    # Colonnes de métadonnées à conserver
    meta_cols = ["PRIMAL_KEY","date"]
    df_meta   = df_sentences[meta_cols].reset_index(drop=True)

    # Colonnes BERT : une par dimension
    bert_cols = [f"bert_{i}" for i in range(embeddings.shape[1])]
    df_bert   = pd.DataFrame(embeddings, columns=bert_cols)

    # Fusion côte à côte
    df_final = pd.concat([df_meta, df_bert], axis=1)
    return df_final


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embedding BERT → parquet colonnes bert_0...bert_767")
    parser.add_argument("--input",      required=True,                      help="Parquet d'entrée (ex: outputs/sentences.parquet)")
    parser.add_argument("--output",     default="outputs/embeddings.parquet", help="Parquet de sortie")
    parser.add_argument("--model",      default=BERT_MODEL,                 help="Modèle BERT de HuggingFace")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int,       help="Taille des batchs")
    args = parser.parse_args()

    # ── Chargement ──
    print(f"\n📂 Chargement de : {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   {len(df)} phrases chargées\n")

    # ── Encodage ──
    encoder    = BertEmbedder(model_name=args.model)
    sentences  = df["sentence"].tolist()
    embeddings = encoder.encode(sentences, batch_size=args.batch_size)

    # ── Construction du DataFrame final ──
    df_final = build_embedding_dataframe(df, embeddings)

    # ── Sauvegarde ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    print(f"\n💾 Parquet sauvegardé → {output_path}")
    print(f"   Shape : {df_final.shape}  ({len(df_final)} phrases × {df_final.shape[1]} colonnes)")
    print(f"\nAperçu des colonnes : {list(df_final.columns[:8])} ... {list(df_final.columns[-3:])}\n")


if __name__ == "__main__":
    main()