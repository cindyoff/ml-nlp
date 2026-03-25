"""
Découpage fichiers txt + construction d'un dataframe fichier parquet
"""

import re
import argparse
from pathlib import Path
from .config import ENCODING, NLTK_LANGUAGE
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# filtrage ligne par ligne 
# en-tête + pied de page
_ADMIN_EXPLICIT = re.compile(
    r'sciences\s*po'
    r'|cevipof'
    r'|imprimerie'
    r'|vu,?\s*le\s+candidat'
    r'|suppléante?\s*:'
    r'|remplaçant\s+éventuel'
    r'|fonds\s+cevipof'
    r'|[eé]lections?\s+(?:l[eé]gislatives?|pr[eé]sidentielles?)'
    r'|\d+(?:e|eme|ème|er|re)?\s*(?:circonscription|arrondissement)'
    r'|rassemblement\s+pour\s+la\s+r[eé]publique'
    r'|union\s+pour\s+une\s+nouvelle\s+majorit',
    re.IGNORECASE,
)

# biographie
_ADMIN_BIO = re.compile(
    r'\b\d{2}\s+ans\b'
    r'|^\s*[A-ZÉÈÀÊ][a-zéèàê]+\s+[A-ZÉÈÀÊ]{2,}', # prenom, nom
    re.MULTILINE,
)

# verbes
_VERB_ENDINGS = re.compile(
    r'\b\w+(?:ons|ez|ais|ait|aient|ions|iez|erai|eras|era|erons|erez|eront'
    r'|âmes|âtes|èrent)\b'             # formes verbales non ambiguës
    r'|\b(?:est|sont|était|étaient|sera|seront|ont|avons|avez|avoir|être)\b',  # auxiliaires
    re.IGNORECASE,
)

MIN_WORDS = 6          # sous six mots, phrase doit avoir un verbe pour être conservée
MIN_SENTENCE_WORDS = 4 # après tokenisation, phrases de moins de 4 mots éliminées


def _is_administrative(line: str) -> bool:
    """
    True si admin
    Critères progressifs : 
      1. Marqueur explicite (Sciences Po, imprimerie…) : admin
      2. Ressemble à une ligne biographique (âge, nom seul) : admin
      3. Courte + sans aucun verbe conjugué : admin
    """
    stripped = line.strip()
    if not stripped:
        return True
    if _ADMIN_EXPLICIT.search(stripped):
        return True
    if _ADMIN_BIO.search(stripped):
        return True
    words = stripped.split()
    if len(words) < MIN_WORDS and not _VERB_ENDINGS.search(stripped):
        return True
    return False

# pré-traitements
def clean_text(text: str) -> tuple[str, float]:
    """
    Texte nettoyé (artefacts, normalise espaces), toutes lignes conservées
    """
    lines   = text.splitlines()
    n_total = sum(1 for l in lines if l.strip())
    n_admin = sum(1 for l in lines if l.strip() and _is_administrative(l))

    filter_ratio = round(n_admin / n_total, 4) if n_total > 0 else 0.0

    text = ' '.join(l for l in lines if l.strip())
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, filter_ratio

# decoupage phrases + tokenisation
def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in sent_tokenize(text, language=NLTK_LANGUAGE)
        if s.strip() and len(s.split()) >= MIN_SENTENCE_WORDS
    ]

# dataframe construction
def build_dataframe(data_dir: str, max_sentences: int = 0) -> pd.DataFrame:
    """
    dataframe avec : 
      - PRIMARY_KEY
      - doc_id
      - date
      - classe : législative, présidentielle
      - sentence
      - filter_ratio : admin détectées
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {data_dir}")

    rows = []
    stopped_early = False

    for class_dir in sorted(data_path.glob("*/*")):
        if stopped_early:
            break
        if not class_dir.is_dir():
            continue

        date_name  = class_dir.parent.name
        class_name = class_dir.name

        txt_files = sorted(class_dir.glob("*.txt"))

        dir_ratios = []
        for txt_file in txt_files:
            try:
                raw  = txt_file.read_text(encoding=ENCODING)
                text, filter_ratio = clean_text(raw)
                dir_ratios.append(filter_ratio)

                if not text:
                    continue

                sentences = split_sentences(text)

                # arrêt si limite est dépassée
                if max_sentences > 0 and len(rows) + len(sentences) > max_sentences:
                    stopped_early = True
                    break

                for j, sentence in enumerate(sentences):
                    rows.append({
                        "doc_id"       : txt_file.stem,
                        "PRIMARY_KEY"  : txt_file.stem+f"_{j}",
                        "date"         : date_name,
                        "classe"       : class_name,
                        "sentence"     : sentence,
                        "filter_ratio" : filter_ratio,
                    })

            except Exception as e:
                print(f"Impossible de lire {txt_file.name} : {e}")

        # stats du répertoire
        if dir_ratios:
            mean_r = sum(dir_ratios) / len(dir_ratios)
            print(
                f"  [{date_name}/{class_name}] {len(txt_files)} fichier(s) — "
                f"ratio filtré moy {mean_r:.1%}  "
                f"(min {min(dir_ratios):.1%} / max {max(dir_ratios):.1%})"
            )
        if stopped_early:
            print(f"\nLimite de {max_sentences} phrases atteinte — arrêt après {len(rows)} phrases")

    df = pd.DataFrame(rows, columns=["doc_id", "PRIMARY_KEY", "date", "classe", "sentence", "filter_ratio"])

    # Résumé global
    if not df.empty:
        mean_global = df.drop_duplicates("doc_id")["filter_ratio"].mean()
        high_noise  = (df.drop_duplicates("doc_id")["filter_ratio"] > 0.5).sum()
        print(
            f"\n{len(df)} phrases extraites depuis {df['doc_id'].nunique()} documents\n"
            f"Ratio de filtrage moyen : {mean_global:.1%}  "
            f"({high_noise} documents avec > 50% de lignes filtrées)\n"
        )
    else:
        print(f"\n{len(df)} phrases extraites depuis 0 documents\n")
    return df

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(description="Tokenisation des discours en phrases")
    parser.add_argument("--data_dir",      required=True,  help="Dossier racine des TXT (ex: text_files/)")
    parser.add_argument("--output",        default="sentences.parquet", help="Fichier parquet de sortie")
    parser.add_argument("--max_sentences", default=0, type=int,
                        help="Limite de phrases (0 = pas de limite). Les documents sont conservés entiers.")
    args = parser.parse_args()

    print(f"\nLecture des fichiers depuis : {args.data_dir}\n")
    df = build_dataframe(args.data_dir, max_sentences=args.max_sentences)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"DataFrame sauvegardé : {output_path}")
    print(f"Shape : {df.shape}")
    print(f"\n{df.head(3).to_string()}\n")

if __name__ == "__main__":
    main()