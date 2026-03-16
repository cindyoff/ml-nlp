"""
tests/test_sentences.py
=======================
Tests pour pipeline/sentences.py :
  - _is_administrative  : détection des lignes admin
  - clean_text          : nettoyage HTML + calcul filter_ratio
  - split_sentences     : filtrage des phrases trop courtes
  - build_dataframe     : output complet sur répertoire tmp
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.sentences import (
    _is_administrative,
    clean_text,
    split_sentences,
    build_dataframe,
)


# ── _is_administrative ────────────────────────────────────────────────────────

class TestIsAdministrative:
    def test_sciences_po_detected(self):
        assert _is_administrative("Sciences Po / fonds CEVIPOF") is True

    def test_cevipof_detected(self):
        assert _is_administrative("Archives CEVIPOF 1988") is True

    def test_imprimerie_detected(self):
        assert _is_administrative("Imprimerie Dupont, Paris 15e") is True

    def test_circonscription_detected(self):
        assert _is_administrative("2ème Circonscription de Paris") is True

    def test_age_bio_detected(self):
        assert _is_administrative("44 ans, marié, trois enfants") is True

    def test_empty_line_detected(self):
        assert _is_administrative("   ") is True

    def test_short_line_no_verb_detected(self):
        # Moins de 6 mots, sans verbe → admin
        assert _is_administrative("Mon programme politique") is True

    def test_normal_sentence_kept(self):
        assert _is_administrative(
            "Nous allons construire une société plus juste et plus égalitaire."
        ) is False

    def test_sentence_with_verb_kept(self):
        assert _is_administrative("La France doit avancer vers l'Europe.") is False

    def test_elections_keyword_detected(self):
        assert _is_administrative("Élections législatives de 1981") is True


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_html_tags(self):
        text, _ = clean_text("Bonjour <b>monde</b>, comment allez-vous aujourd'hui?")
        assert "<b>" not in text
        assert "monde" in text

    def test_removes_brackets(self):
        text, _ = clean_text("Le président parle. [Applaudissements] C'est important.")
        assert "[Applaudissements]" not in text

    def test_filter_ratio_zero_for_clean_text(self):
        # Texte entièrement normal → ratio bas (toutes les lignes ont un verbe)
        text = (
            "Nous construisons une France meilleure.\n"
            "Les citoyens méritent un avenir radieux.\n"
            "Nous travaillons pour vous chaque jour."
        )
        _, ratio = clean_text(text)
        assert 0.0 <= ratio <= 1.0

    def test_filter_ratio_high_for_admin_text(self):
        # Texte plein de lignes admin
        text = (
            "Sciences Po\n"
            "Imprimerie Nationale\n"
            "44 ans, marié\n"
            "CEVIPOF archives\n"
            "2ème Circonscription\n"
        )
        _, ratio = clean_text(text)
        assert ratio > 0.5

    def test_returns_tuple(self):
        result = clean_text("Texte simple pour tester.")
        assert isinstance(result, tuple) and len(result) == 2

    def test_empty_text(self):
        text, ratio = clean_text("")
        assert text == ""
        assert ratio == 0.0

    def test_normalizes_whitespace(self):
        text, _ = clean_text("Mot1   \n\n  Mot2\t\tMot3")
        assert "  " not in text  # pas d'espaces doubles


# ── split_sentences ───────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_filters_short_sentences(self):
        # Phrases de moins de 4 mots doivent être filtrées
        text = "Oui. Non. Je veux une France plus juste et solidaire pour tous."
        sentences = split_sentences(text)
        for s in sentences:
            assert len(s.split()) >= 4

    def test_normal_sentences_kept(self):
        text = (
            "Nous voulons construire une société meilleure pour tous les citoyens français. "
            "L'éducation est la priorité absolue de notre programme politique."
        )
        sentences = split_sentences(text)
        assert len(sentences) >= 1

    def test_returns_list(self):
        result = split_sentences("Une phrase normale avec des mots.")
        assert isinstance(result, list)

    def test_empty_text(self):
        result = split_sentences("")
        assert result == []

    def test_strips_whitespace(self):
        sentences = split_sentences("  Voici une phrase avec des espaces inutiles autour.  ")
        for s in sentences:
            assert s == s.strip()


# ── build_dataframe ───────────────────────────────────────────────────────────

class TestBuildDataframe:
    def _make_txt_dir(self, tmp_path: Path, n_files: int = 3) -> Path:
        """Crée une structure data_dir/1981/legislatives/*.txt avec du contenu fictif."""
        data_dir = tmp_path / "text_files"
        doc_dir  = data_dir / "1981" / "legislatives"
        doc_dir.mkdir(parents=True)

        for i in range(n_files):
            (doc_dir / f"doc_{i}.txt").write_text(
                f"Nous allons construire une France plus juste et plus solidaire pour tous.\n"
                f"L'emploi est la priorité numéro un de notre programme politique ambitieux.\n"
                f"Nous nous engageons à défendre les droits de chaque citoyen français.\n"
                f"Notre vision pour l'avenir repose sur la justice sociale et économique.\n",
                encoding="utf-8",
            )
        return data_dir

    def test_output_has_required_columns(self, tmp_path):
        """Le DataFrame produit contient toutes les colonnes attendues."""
        data_dir = self._make_txt_dir(tmp_path)
        df = build_dataframe(str(data_dir))
        required = {"PRIMARY_KEY", "doc_id", "date", "classe", "sentence", "filter_ratio"}
        assert required.issubset(set(df.columns))

    def test_primary_key_is_unique(self, tmp_path):
        """Chaque phrase a un PRIMARY_KEY unique."""
        data_dir = self._make_txt_dir(tmp_path, n_files=3)
        df = build_dataframe(str(data_dir))
        assert df["PRIMARY_KEY"].nunique() == len(df)

    def test_primary_key_format(self, tmp_path):
        """PRIMARY_KEY est au format '{doc_id}_{index}'."""
        data_dir = self._make_txt_dir(tmp_path, n_files=1)
        df = build_dataframe(str(data_dir))
        for _, row in df.iterrows():
            assert row["PRIMARY_KEY"].startswith(row["doc_id"])
            parts = row["PRIMARY_KEY"].split("_")
            assert parts[-1].isdigit()

    def test_date_and_classe_extracted(self, tmp_path):
        """date et classe sont extraits depuis la structure de dossiers."""
        data_dir = self._make_txt_dir(tmp_path)
        df = build_dataframe(str(data_dir))
        assert (df["date"] == "1981").all()
        assert (df["classe"] == "legislatives").all()

    def test_filter_ratio_between_0_and_1(self, tmp_path):
        """filter_ratio est toujours dans [0, 1]."""
        data_dir = self._make_txt_dir(tmp_path)
        df = build_dataframe(str(data_dir))
        assert (df["filter_ratio"] >= 0.0).all()
        assert (df["filter_ratio"] <= 1.0).all()

    def test_max_sentences_respected(self, tmp_path):
        """max_sentences limite le nombre de phrases (documents complets)."""
        data_dir = self._make_txt_dir(tmp_path, n_files=10)
        df_full    = build_dataframe(str(data_dir))
        df_limited = build_dataframe(str(data_dir), max_sentences=5)
        assert len(df_limited) <= len(df_full)

    def test_missing_dir_raises(self, tmp_path):
        """FileNotFoundError si le répertoire n'existe pas."""
        with pytest.raises(FileNotFoundError):
            build_dataframe(str(tmp_path / "inexistant"))

    def test_empty_dir_returns_empty_df(self, tmp_path):
        """DataFrame vide si aucun fichier .txt trouvé."""
        empty_dir = tmp_path / "vide"
        empty_dir.mkdir()
        df = build_dataframe(str(empty_dir))
        assert len(df) == 0

    def test_sentences_min_word_count(self, tmp_path):
        """Toutes les phrases ont au moins 4 mots (filtrage MIN_SENTENCE_WORDS)."""
        data_dir = self._make_txt_dir(tmp_path)
        df = build_dataframe(str(data_dir))
        for sentence in df["sentence"]:
            assert len(sentence.split()) >= 4

    def test_output_saved_to_parquet(self, tmp_path):
        """Le parquet sauvegardé est lisible et a le bon schéma."""
        data_dir  = self._make_txt_dir(tmp_path)
        out_path  = tmp_path / "sentences.parquet"
        df = build_dataframe(str(data_dir))
        df.to_parquet(out_path, index=False)

        df_reloaded = pd.read_parquet(out_path)
        assert set(df_reloaded.columns) == set(df.columns)
        assert len(df_reloaded) == len(df)
