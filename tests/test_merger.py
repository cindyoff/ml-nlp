"""
Tests unitaires pour merger.py :
  - merge happy path
  - clés manquantes (warning seulement, pas d'erreur)
  - colonne embedding présente en sortie
  - validation de schéma
"""

import numpy as np
import pandas as pd
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.merger import merge
from pipeline.config import validate_schema


def _make_embeddings(n=10) -> pd.DataFrame:
    return pd.DataFrame({
        "PRIMARY_KEY": [f"doc_{i}" for i in range(n)],
        "doc_id"     : [f"doc_{i}" for i in range(n)],
        "date"       : ["1981"] * n,
        "classe"     : ["legislatives"] * n,
        "embedding"  : [np.random.rand(768).astype(np.float32).tolist() for _ in range(n)],
    })


def _make_features(n=10) -> pd.DataFrame:
    return pd.DataFrame({
        "PRIMARY_KEY"  : [f"doc_{i}" for i in range(n)],
        "doc_id"       : [f"doc_{i}" for i in range(n)],
        "date"         : ["1981"] * n,
        "classe"       : ["legislatives"] * n,
        "sentence"     : [f"phrase {i}" for i in range(n)],
        "filter_ratio" : np.random.rand(n),
        "n_digits"     : np.random.randint(0, 5, n),
    })


class TestMerge:
    def test_happy_path(self, tmp_path):
        """Fusion normale : output contient les colonnes des deux sources en même temps"""
        emb_path  = tmp_path / "embeddings.parquet"
        feat_path = tmp_path / "features.parquet"
        out_path  = tmp_path / "final.parquet"

        _make_embeddings(10).to_parquet(emb_path, index=False)
        _make_features(10).to_parquet(feat_path, index=False)

        merge(str(emb_path), str(feat_path), str(out_path))

        df = pd.read_parquet(out_path)
        assert "embedding" in df.columns
        assert "sentence" in df.columns
        assert len(df) == 10

    def test_output_has_embedding_col(self, tmp_path):
        """La colonne 'embedding' est présente dans fichier final.parquet"""
        emb_path  = tmp_path / "embeddings.parquet"
        feat_path = tmp_path / "features.parquet"
        out_path  = tmp_path / "final.parquet"

        _make_embeddings(5).to_parquet(emb_path, index=False)
        _make_features(5).to_parquet(feat_path, index=False)

        merge(str(emb_path), str(feat_path), str(out_path))

        df = pd.read_parquet(out_path)
        assert "embedding" in df.columns

    def test_missing_keys_does_not_raise(self, tmp_path, capsys):
        """Clés manquantes dans features : warning affiché, pas d'exception"""
        emb_path  = tmp_path / "embeddings.parquet"
        feat_path = tmp_path / "features.parquet"
        out_path  = tmp_path / "final.parquet"

        emb  = _make_embeddings(10)
        feat = _make_features(8)   # 2 clés manquantes dans features

        emb.to_parquet(emb_path, index=False)
        feat.to_parquet(feat_path, index=False)

        merge(str(emb_path), str(feat_path), str(out_path))  # ne doit pas lever

        captured = capsys.readouterr()
        assert "clés présentes dans embeddings" in captured.out

    def test_schema_validation_raises_missing_embedding(self, tmp_path):
        """validate_schema lève ValueError si la colonne 'embedding' est absente"""
        df = pd.DataFrame({"PRIMARY_KEY": ["a", "b"]})
        with pytest.raises(ValueError, match="embedding"):
            validate_schema(df, {"PRIMARY_KEY", "embedding"}, "embeddings.parquet")

    def test_output_shape(self, tmp_path):
        """Le nombre de lignes en sortie correspond à features (jointure gauche sur features)"""
        emb_path  = tmp_path / "embeddings.parquet"
        feat_path = tmp_path / "features.parquet"
        out_path  = tmp_path / "final.parquet"

        _make_embeddings(10).to_parquet(emb_path, index=False)
        _make_features(10).to_parquet(feat_path, index=False)

        merge(str(emb_path), str(feat_path), str(out_path))

        df = pd.read_parquet(out_path)
        assert len(df) == 10
