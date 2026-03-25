from pathlib import Path

def load_lexicon(path: str) -> set[str]:
    """
    Lecture fichier txt et retour d'un set
    Ignore les lignes vides et commentaires
    """
    return {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

def load_lexicons(*paths: str) -> set[str]:
    """Fusion de plusieurs txt en un seul set"""
    result = set()
    for path in paths:
        result |= load_lexicon(path)
    return result