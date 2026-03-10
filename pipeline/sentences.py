"""
sentences.py
============
Découpe les fichiers TXT en phrases et construit un DataFrame
sauvegardé en parquet.

Structure attendue :
    data_dir/
        date_*/
            legislatives/
                *.txt
            presidentielle/
                *.txt

Usage :
    python -m pipeline.sentences --data_dir text_files/ --output sentences.parquet
"""

import re
import argparse
from pathlib import Path
from config import ENCODING, NLTK_LANGUAGE 
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# ──────────────────────────────────────────────
# PRÉTRAITEMENT LÉGER (artefacts SQL/HTML)
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)        # balises HTML
    text = re.sub(r'\[.*?\]', '', text)        # [Applaudissements] etc.
    text = re.sub(r'\s+', ' ', text)           # espaces multiples / \n
    return text


# ──────────────────────────────────────────────
# DÉCOUPAGE EN PHRASES + TOKENISATION
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text, language=NLTK_LANGUAGE) if s.strip()]




# ──────────────────────────────────────────────
# CONSTRUCTION DU DATAFRAME
# ──────────────────────────────────────────────

def build_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Parcourt data_dir/*/classe/*.txt et retourne un DataFrame avec :
      - PRIMARY_KEY : sentence ID
      - doc_id    : nom du fichier source
      - date      : nom du dossier date intermédiaire
      - classe    : legislatives | presidentielle
      - sentence  : texte de la phrase
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {data_dir}")

    rows = []

    # Descend 2 niveaux : date/ puis classe/
    for class_dir in sorted(data_path.glob("*/*")):
        if not class_dir.is_dir():
            continue

        date_name  = class_dir.parent.name
        class_name = class_dir.name

        txt_files = sorted(class_dir.glob("*.txt"))
        print(f"  [{date_name}/{class_name}] {len(txt_files)} fichier(s)")

        for txt_file in txt_files:
            try:
                raw  = txt_file.read_text(encoding=ENCODING)
                text = clean_text(raw)

                if not text:
                    continue

                sentences = split_sentences(text)

                for j, sentence in enumerate(sentences):
                    rows.append({
                        "doc_id"   : txt_file.stem,
                        "PRIMAL_KEY" : txt_file.stem+f"_{j}",
                        "date"     : date_name,
                        "classe"   : class_name,
                        "sentence" : sentence                    })

            except Exception as e:
                print(f"  ⚠  Impossible de lire {txt_file.name} : {e}")

    df = pd.DataFrame(rows, columns=["doc_id", "PRIMAL_KEY", "date", "classe", "sentence"])
    print(f"\n✅ {len(df)} phrases extraites depuis {df['doc_id'].nunique()} documents\n")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(description="Tokenisation des discours en phrases")
    parser.add_argument("--data_dir", required=True, help="Dossier racine des TXT (ex: text_files/)")
    parser.add_argument("--output",   default="sentences.parquet", help="Fichier parquet de sortie")
    args = parser.parse_args()

    print(f"\n📂 Lecture des fichiers depuis : {args.data_dir}\n")
    df = build_dataframe(args.data_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 DataFrame sauvegardé → {output_path}")
    print(f"   Shape : {df.shape}")
    print(f"\n{df.head(3).to_string()}\n")


if __name__ == "__main__":
    main()