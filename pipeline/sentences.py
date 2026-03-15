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
from .config import ENCODING, NLTK_LANGUAGE
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# ──────────────────────────────────────────────
# PATTERNS ADMINISTRATIFS (filtrés ligne par ligne)
# ──────────────────────────────────────────────

# Marqueurs explicites d'en-têtes / pieds de page
_ADMIN_EXPLICIT = re.compile(
    r'sciences\s*po'                                        # "Sciences Po / fonds CEVIPOF"
    r'|cevipof'
    r'|imprimerie'                                          # info d'impression
    r'|vu,?\s*le\s+candidat'                               # tampon administratif
    r'|suppléante?\s*:'                                     # info suppléant (m. et f.)
    r'|remplaçant\s+éventuel'                              # synonyme de suppléant
    r'|fonds\s+cevipof'
    r'|[eé]lections?\s+(?:l[eé]gislatives?|pr[eé]sidentielles?)'  # couvre accents ET majuscules
    r'|\d+(?:e|eme|ème|er|re)?\s*(?:circonscription|arrondissement)'  # "2eme Circonscription"
    r'|rassemblement\s+pour\s+la\s+r[eé]publique'          # sigle RPR
    r'|union\s+pour\s+une\s+nouvelle\s+majorit',            # sigle UNM
    re.IGNORECASE,
)

# Lignes biographiques sans contenu politique :
#   "44 ans, marié" / "36 ans, célibataire, 3 enfants" / "37 ans, informaticien"
_ADMIN_BIO = re.compile(
    r'\b\d{2}\s+ans\b'                                     # âge n'importe où dans la ligne
    r'|^\s*[A-ZÉÈÀÊ][a-zéèàê]+\s+[A-ZÉÈÀÊ]{2,}',         # "Prénom NOM" seul (début de ligne)
    re.MULTILINE,
)

# Verbes conjugués français — exclut les faux positifs en "-ment" (noms)
# On évite "ent" seul (trop ambigu avec "rassemblement", "gouvernement"…)
_VERB_ENDINGS = re.compile(
    r'\b\w+(?:ons|ez|ais|ait|aient|ions|iez|erai|eras|era|erons|erez|eront'
    r'|âmes|âtes|èrent)\b'             # formes verbales non ambiguës
    r'|\b(?:est|sont|était|étaient|sera|seront|ont|avons|avez|avoir|être)\b',  # auxiliaires
    re.IGNORECASE,
)

MIN_WORDS = 6          # en dessous de ce seuil, la ligne doit contenir un verbe pour être gardée
MIN_SENTENCE_WORDS = 4 # après tokenisation, phrases de moins de 4 mots éliminées


def _is_administrative(line: str) -> bool:
    """
    Retourne True si la ligne est probablement du méta-contenu (en-tête, pied de page,
    info biographique) plutôt que du discours politique réel.

    Logique en cascade :
      1. Contient un marqueur explicite (Sciences Po, imprimerie…) → admin
      2. Ressemble à une ligne biographique (âge, nom seul) → admin
      3. Courte ET sans aucun verbe conjugué → admin
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


# ──────────────────────────────────────────────
# PRÉTRAITEMENT (artefacts SQL/HTML + lignes admin)
# ──────────────────────────────────────────────

def clean_text(text: str) -> tuple[str, float]:
    """
    Retourne (texte nettoyé, filter_ratio).

    Nettoie les artefacts HTML/brackets et normalise les espaces.
    Toutes les lignes sont conservées — aucun filtrage de contenu.
    filter_ratio = proportion de lignes non vides détectées comme administratives
    (stocké comme feature documentaire, utilisé à l'entraînement).
    """
    lines   = text.splitlines()
    n_total = sum(1 for l in lines if l.strip())
    n_admin = sum(1 for l in lines if l.strip() and _is_administrative(l))

    filter_ratio = round(n_admin / n_total, 4) if n_total > 0 else 0.0

    text = ' '.join(l for l in lines if l.strip())
    text = re.sub(r'<[^>]+>', '', text)        # balises HTML résiduelles
    text = re.sub(r'\[.*?\]', '', text)        # [Applaudissements] etc.
    text = re.sub(r'\s+', ' ', text).strip()   # espaces multiples / \n
    return text, filter_ratio


# ──────────────────────────────────────────────
# DÉCOUPAGE EN PHRASES + TOKENISATION
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in sent_tokenize(text, language=NLTK_LANGUAGE)
        if s.strip() and len(s.split()) >= MIN_SENTENCE_WORDS
    ]




# ──────────────────────────────────────────────
# CONSTRUCTION DU DATAFRAME
# ──────────────────────────────────────────────

def build_dataframe(data_dir: str, max_sentences: int = 0) -> pd.DataFrame:
    """
    Parcourt data_dir/*/classe/*.txt et retourne un DataFrame avec :
      - PRIMARY_KEY  : sentence ID
      - doc_id       : nom du fichier source
      - date         : nom du dossier date intermédiaire
      - classe       : legislatives | presidentielle
      - sentence     : texte de la phrase
      - filter_ratio : proportion de lignes détectées comme administratives (feature)

    Toutes les phrases sont conservées. La comparaison 2-classes / 3-classes
    se fait à l'entraînement en filtrant sur le label "autre".

    Si max_sentences > 0, les documents sont ajoutés en entier jusqu'à ce que
    la limite soit atteinte (aucun document n'est tronqué).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {data_dir}")

    rows = []
    stopped_early = False

    # Descend 2 niveaux : date/ puis classe/
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

                # Si ajouter ce document dépasse la limite, on s'arrête
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
                print(f"  ⚠  Impossible de lire {txt_file.name} : {e}")

        # Statistiques du répertoire
        if dir_ratios:
            mean_r = sum(dir_ratios) / len(dir_ratios)
            print(
                f"  [{date_name}/{class_name}] {len(txt_files)} fichier(s) — "
                f"ratio filtré moy {mean_r:.1%}  "
                f"(min {min(dir_ratios):.1%} / max {max(dir_ratios):.1%})"
            )
        if stopped_early:
            print(f"\n  ⏹  Limite de {max_sentences} phrases atteinte — arrêt après {len(rows)} phrases")

    df = pd.DataFrame(rows, columns=["doc_id", "PRIMARY_KEY", "date", "classe", "sentence", "filter_ratio"])

    # Résumé global
    if not df.empty:
        mean_global = df.drop_duplicates("doc_id")["filter_ratio"].mean()
        high_noise  = (df.drop_duplicates("doc_id")["filter_ratio"] > 0.5).sum()
        print(
            f"\n✅ {len(df)} phrases extraites depuis {df['doc_id'].nunique()} documents\n"
            f"   Ratio de filtrage moyen : {mean_global:.1%}  "
            f"({high_noise} documents avec > 50% de lignes filtrées)\n"
        )
    else:
        print(f"\n✅ {len(df)} phrases extraites depuis 0 documents\n")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(description="Tokenisation des discours en phrases")
    parser.add_argument("--data_dir",      required=True,  help="Dossier racine des TXT (ex: text_files/)")
    parser.add_argument("--output",        default="sentences.parquet", help="Fichier parquet de sortie")
    parser.add_argument("--max_sentences", default=0, type=int,
                        help="Limite de phrases (0 = pas de limite). Les documents sont conservés entiers.")
    args = parser.parse_args()

    print(f"\n📂 Lecture des fichiers depuis : {args.data_dir}\n")
    df = build_dataframe(args.data_dir, max_sentences=args.max_sentences)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 DataFrame sauvegardé → {output_path}")
    print(f"   Shape : {df.shape}")
    print(f"\n{df.head(3).to_string()}\n")


if __name__ == "__main__":
    main()