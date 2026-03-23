"""
Tests unitaires pour pipeline/labeler.py :
  - apply_labels path
  - label invalide : ValueError
  - colonne PRIMARY_KEY absente dans CSV : ValueError
  - fichier 2 classes bien crée
  - phrases sans label : NaN conservé
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.labeler import apply_labels


def _make_parquet(tmp_path, n=20) -> Path:
    df = pd.DataFrame({
        "PRIMARY_KEY"  : [f"doc_0_{i}" for i in range(n)],
        "doc_id"       : ["doc_0"] * n,
        "date"         : ["1981"] * n,
        "classe"       : ["legislatives"] * n,
        "sentence"     : [f"phrase {i}" for i in range(n)],
        "filter_ratio" : np.zeros(n),
        "embedding"    : [np.zeros(768).tolist() for _ in range(n)],
    })
    path = tmp_path / "final.parquet"
    df.to_parquet(path, index=False)
    return path


def _make_csv(tmp_path, labels: dict) -> Path:
    rows = [{"PRIMARY_KEY": k, "label": v} for k, v in labels.items()]
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestApplyLabels:
    def test_happy_path(self, tmp_path):
        """Labels bien appliqués sur les PRIMARY_KEY correspondantes"""
        parquet = _make_parquet(tmp_path, n=10)
        csv     = _make_csv(tmp_path, {
            "doc_0_0": "langue_de_bois",
            "doc_0_1": "non_langue_de_bois",
            "doc_0_2": "langue_de_bois",
        })
        out = tmp_path / "final_labeled.parquet"
        apply_labels(str(csv), str(parquet), str(out))

        df = pd.read_parquet(out)
        assert df.loc[df["PRIMARY_KEY"] == "doc_0_0", "label"].iloc[0] == "langue_de_bois"
        assert df.loc[df["PRIMARY_KEY"] == "doc_0_1", "label"].iloc[0] == "non_langue_de_bois"

    def test_unlabeled_rows_are_nan(self, tmp_path):
        """Phrases sans label doivent avoir NaN dans la colonne nommée label"""
        parquet = _make_parquet(tmp_path, n=10)
        csv     = _make_csv(tmp_path, {"doc_0_0": "langue_de_bois"})
        out = tmp_path / "final_labeled.parquet"
        apply_labels(str(csv), str(parquet), str(out))

        df = pd.read_parquet(out)
        unlabeled = df[df["PRIMARY_KEY"] != "doc_0_0"]["label"]
        assert unlabeled.isna().all()

    def test_invalid_label_raises(self, tmp_path):
        """Label inconnu lève ValueError"""
        parquet = _make_parquet(tmp_path, n=5)
        csv     = _make_csv(tmp_path, {"doc_0_0": "INCONNU"})
        out = tmp_path / "final_labeled.parquet"
        with pytest.raises(ValueError, match="Labels inconnus"):
            apply_labels(str(csv), str(parquet), str(out))

    def test_missing_primary_key_col_in_csv(self, tmp_path):
        """CSV sans colonne PRIMARY_KEY : ValueError"""
        parquet = _make_parquet(tmp_path, n=5)
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"id": ["x"], "label": ["langue_de_bois"]}).to_csv(bad_csv, index=False)
        out = tmp_path / "labeled.parquet"
        with pytest.raises(ValueError, match="PRIMARY_KEY"):
            apply_labels(str(bad_csv), str(parquet), str(out))

    def test_produces_2class_file(self, tmp_path):
        """Le fichier 2 classes est créé en parallèle du fichier principal"""
        parquet = _make_parquet(tmp_path, n=10)
        csv     = _make_csv(tmp_path, {
            "doc_0_0": "langue_de_bois",
            "doc_0_1": "autre",
        })
        out = tmp_path / "final_labeled.parquet"
        apply_labels(str(csv), str(parquet), str(out))

        out_2class = tmp_path / "final_labeled_2class.parquet"
        assert out_2class.exists()

    def test_2class_excludes_autre(self, tmp_path):
        """Le fichier 2 classes ne contient pas les phrases labellisées comme autre"""
        parquet = _make_parquet(tmp_path, n=10)
        csv     = _make_csv(tmp_path, {
            "doc_0_0": "langue_de_bois",
            "doc_0_1": "autre",
            "doc_0_2": "non_langue_de_bois",
        })
        out = tmp_path / "final_labeled.parquet"
        apply_labels(str(csv), str(parquet), str(out))

        df_2class = pd.read_parquet(tmp_path / "final_labeled_2class.parquet")
        assert "autre" not in df_2class["label"].dropna().values

    def test_csv_not_found_raises(self, tmp_path):
        """CSV introuvable : FileNotFoundError"""
        parquet = _make_parquet(tmp_path, n=5)
        with pytest.raises(FileNotFoundError):
            apply_labels(str(tmp_path / "missing.csv"), str(parquet), str(tmp_path / "out.parquet"))
