from pathlib import Path

def load_lexicon(path: str) -> set[str]:
    """
    Lit un fichier txt (un mot/expression par ligne) et retourne un set.
    Ignore les lignes vides et les commentaires (#).
    """
    return {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

def load_lexicons(*paths: str) -> set[str]:
    """Fusionne plusieurs fichiers txt en un seul set."""
    result = set()
    for path in paths:
        result |= load_lexicon(path)
    return result