"""
tests/test_modelisation.py
==========================
Tests unitaires pour les fonctions de modelisation.py :
  - compute_iv
  - select_by_iv
  - calibrate_threshold
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modelisation import compute_iv, select_by_iv, calibrate_threshold


# ── compute_iv ────────────────────────────────────────────────────────────────

class TestComputeIV:
    def test_strong_separation(self):
        """IV élevé quand la feature discrimine bien les classes (gaussiennes décalées avec overlap)."""
        np.random.seed(42)
        n = 500
        # Gaussiennes décalées — overlap suffisant pour que tous les bins aient des events des 2 classes
        x = pd.Series(np.concatenate([
            np.random.normal(-2, 1, n // 2),
            np.random.normal( 2, 1, n // 2),
        ]))
        y = pd.Series(np.array([0] * (n // 2) + [1] * (n // 2)))
        iv = compute_iv(x, y)
        assert iv > 0.5, f"IV attendu > 0.5 pour séparation forte, obtenu {iv:.4f}"

    def test_no_variation(self):
        """IV = 0 quand la feature est constante."""
        x = pd.Series([1.0] * 100)
        y = pd.Series([0] * 50 + [1] * 50)
        iv = compute_iv(x, y)
        assert iv == 0.0, f"IV attendu = 0 pour feature constante, obtenu {iv}"

    def test_random_noise(self):
        """IV proche de 0 pour une feature aléatoire non corrélée à y."""
        np.random.seed(0)
        x = pd.Series(np.random.randn(1000))
        y = pd.Series(np.random.randint(0, 2, 1000))
        iv = compute_iv(x, y)
        assert iv < 0.1, f"IV attendu < 0.1 pour bruit aléatoire, obtenu {iv:.4f}"

    def test_returns_float(self):
        """compute_iv retourne toujours un float."""
        x = pd.Series(np.linspace(0, 1, 50))
        y = pd.Series([0] * 25 + [1] * 25)
        result = compute_iv(x, y)
        assert isinstance(result, float)

    def test_all_same_class(self):
        """IV = 0.0 si toutes les observations appartiennent à la même classe."""
        x = pd.Series(np.linspace(0, 1, 100))
        y = pd.Series([1] * 100)
        iv = compute_iv(x, y)
        assert iv == 0.0


# ── select_by_iv ──────────────────────────────────────────────────────────────

class TestSelectByIV:
    def _make_df(self):
        np.random.seed(42)
        n = 400
        # Feature forte : gaussiennes décalées avec overlap (IV calculable)
        # Feature faible : bruit pur
        df = pd.DataFrame({
            "strong": np.concatenate([
                np.random.normal(-2, 1, n // 2),
                np.random.normal( 2, 1, n // 2),
            ]),
            "weak"  : np.random.randn(n),
            "y"     : [0] * (n // 2) + [1] * (n // 2),
        })
        return df

    def test_filters_correctly(self):
        """La feature forte est retenue, la feature faible est écartée."""
        df = self._make_df()
        selected, iv_df = select_by_iv(df, ["strong", "weak"], threshold=0.1)
        assert "strong" in selected
        assert "weak" not in selected

    def test_returns_iv_dataframe(self):
        """Le second retour est un DataFrame avec colonne IV."""
        df = self._make_df()
        _, iv_df = select_by_iv(df, ["strong", "weak"], threshold=0.1)
        assert "IV" in iv_df.columns
        assert set(iv_df.index) == {"strong", "weak"}

    def test_empty_if_all_below_threshold(self):
        """Liste vide si aucune feature n'atteint le seuil."""
        np.random.seed(0)
        df = pd.DataFrame({
            "f1": np.random.randn(200),
            "f2": np.random.randn(200),
            "y" : np.random.randint(0, 2, 200),
        })
        selected, _ = select_by_iv(df, ["f1", "f2"], threshold=0.5)
        assert selected == []

    def test_sorted_descending(self):
        """iv_df est trié par IV décroissant."""
        df = self._make_df()
        _, iv_df = select_by_iv(df, ["strong", "weak"], threshold=0.0)
        assert iv_df["IV"].iloc[0] >= iv_df["IV"].iloc[1]


# ── calibrate_threshold ───────────────────────────────────────────────────────

class TestCalibrateThreshold:
    def test_balanced_costs_near_half(self):
        """Avec coûts FP=FN et probabilités bien séparées, le seuil reste raisonnable."""
        np.random.seed(42)
        y    = np.array([0] * 100 + [1] * 100)
        # Probabilités bien séparées
        proba = np.concatenate([
            np.random.uniform(0.0, 0.4, 100),
            np.random.uniform(0.6, 1.0, 100),
        ])
        t, f1 = calibrate_threshold(proba, y)
        assert 0.3 <= t <= 0.7, f"Seuil attendu ≈ 0.5, obtenu {t:.3f}"
        assert f1 > 0.8, f"F1 attendu > 0.8, obtenu {f1:.4f}"

    def test_asymmetric_costs_shift_threshold(self):
        """Coût FN élevé → seuil plus bas (moins de faux négatifs)."""
        np.random.seed(0)
        y    = np.array([0] * 150 + [1] * 50)
        proba = np.concatenate([
            np.random.uniform(0.1, 0.5, 150),
            np.random.uniform(0.4, 0.9, 50),
        ])
        t_balanced,   _ = calibrate_threshold(proba, y, cost_fp=1.0, cost_fn=1.0)
        t_fn_heavy,   _ = calibrate_threshold(proba, y, cost_fp=1.0, cost_fn=5.0)
        assert t_fn_heavy <= t_balanced, (
            f"Seuil avec cost_fn élevé ({t_fn_heavy:.3f}) devrait être ≤ seuil équilibré ({t_balanced:.3f})"
        )

    def test_returns_float_tuple(self):
        """Retourne un tuple (float, float)."""
        y     = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        result = calibrate_threshold(proba, y)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_threshold_in_valid_range(self):
        """Le seuil retourné est toujours dans [0.01, 0.99]."""
        np.random.seed(1)
        y     = np.random.randint(0, 2, 200)
        proba = np.random.rand(200)
        t, _  = calibrate_threshold(proba, y)
        assert 0.01 <= t <= 0.99
