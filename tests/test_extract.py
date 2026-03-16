"""
tests/test_extract.py
=====================
Tests pour pipeline/extract_text.py :
  - index_database  : vérifie que les 3 index SQL sont créés
  - extract_to_txt  : vérifie que les dossiers et fichiers .txt sont produits
  - open (step)     : vérifie que open_database est appelé

Les dépendances arkindex_export (Element, Transcription, list_children, database)
sont mockées — aucune base SQLite réelle n'est requise.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── index_database ────────────────────────────────────────────────────────────

class TestIndexDatabase:
    def test_creates_three_indexes(self):
        """index_database appelle execute_sql pour les 3 CREATE INDEX attendus."""
        mock_db = MagicMock()
        mock_db.is_closed.return_value = False
        mock_db.atomic.return_value.__enter__ = MagicMock(return_value=None)
        mock_db.atomic.return_value.__exit__  = MagicMock(return_value=False)

        with patch("pipeline.extract_text.open_database"), \
             patch("pipeline.extract_text.database", mock_db):
            from pipeline.extract_text import index_database
            index_database(Path("fake.sqlite"))

        sql_calls = [c.args[0] for c in mock_db.execute_sql.call_args_list]
        index_names = [
            "idx_elementpath_parent_child",
            "idx_elementpath_child",
            "idx_element_type",
        ]
        for idx in index_names:
            assert any(idx in sql for sql in sql_calls), \
                f"Index '{idx}' non trouvé dans les appels SQL : {sql_calls}"

    def test_opens_database_when_closed(self):
        """Si la DB est fermée, connect() est appelé."""
        mock_db = MagicMock()
        mock_db.is_closed.return_value = True
        mock_db.atomic.return_value.__enter__ = MagicMock(return_value=None)
        mock_db.atomic.return_value.__exit__  = MagicMock(return_value=False)

        with patch("pipeline.extract_text.open_database"), \
             patch("pipeline.extract_text.database", mock_db):
            from pipeline.extract_text import index_database
            index_database(Path("fake.sqlite"))

        mock_db.connect.assert_called_once()

    def test_closes_database_after_indexing(self):
        """database.close() est toujours appelé en fin d'exécution."""
        mock_db = MagicMock()
        mock_db.is_closed.return_value = False
        mock_db.atomic.return_value.__enter__ = MagicMock(return_value=None)
        mock_db.atomic.return_value.__exit__  = MagicMock(return_value=False)

        with patch("pipeline.extract_text.open_database"), \
             patch("pipeline.extract_text.database", mock_db):
            from pipeline.extract_text import index_database
            index_database(Path("fake.sqlite"))

        mock_db.close.assert_called_once()

    def test_open_database_called_with_path(self):
        """open_database est appelé avec le chemin fourni."""
        mock_db = MagicMock()
        mock_db.is_closed.return_value = False
        mock_db.atomic.return_value.__enter__ = MagicMock(return_value=None)
        mock_db.atomic.return_value.__exit__  = MagicMock(return_value=False)

        db_path = Path("my_database.sqlite")
        with patch("pipeline.extract_text.open_database") as mock_open, \
             patch("pipeline.extract_text.database", mock_db):
            from pipeline.extract_text import index_database
            index_database(db_path)

        mock_open.assert_called_once_with(db_path)


# ── extract_to_txt ────────────────────────────────────────────────────────────

class TestExtractToTxt:
    def _mock_document(self, name: str, pages_text: list[str]) -> MagicMock:
        """Crée un document mock avec ses pages et transcriptions."""
        doc = MagicMock()
        doc.name = name
        doc.id   = f"id_{name}"

        pages = []
        for text in pages_text:
            page = MagicMock()
            page.id = f"page_{name}_{text[:5]}"
            transcription = MagicMock()
            transcription.text = text
            pages.append((page, transcription))
        return doc, pages

    def test_creates_output_folders(self, tmp_path, monkeypatch):
        """extract_to_txt crée les dossiers data/text_files/année/type."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        mock_element  = MagicMock()
        mock_element.select.return_value.where.return_value.count.return_value = 0

        with patch("pipeline.extract_text.Element",      mock_element), \
             patch("pipeline.extract_text.Transcription", MagicMock()), \
             patch("pipeline.extract_text.list_children") as mock_lc:

            mock_lc.return_value.where.return_value.count.return_value = 0
            mock_lc.return_value.where.return_value.__iter__ = MagicMock(return_value=iter([]))

            from pipeline.extract_text import extract_to_txt
            extract_to_txt()

        expected_dirs = [
            tmp_path / "data" / "text_files" / "1981" / "legislatives",
            tmp_path / "data" / "text_files" / "1988" / "legislatives",
            tmp_path / "data" / "text_files" / "1988" / "presidentielle",
            tmp_path / "data" / "text_files" / "1993" / "legislatives",
        ]
        for d in expected_dirs:
            assert d.exists(), f"Dossier attendu absent : {d}"

    def test_writes_txt_files(self, tmp_path, monkeypatch):
        """extract_to_txt écrit un fichier .txt par document avec transcription."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        transcription = MagicMock()
        transcription.text = "Contenu du discours politique."

        page = MagicMock()
        page.id = "page_1"

        document = MagicMock()
        document.name = "EL136_doc_test"
        document.id   = "doc_id_1"

        mock_transcription_cls = MagicMock()
        mock_transcription_cls.select.return_value.where.return_value.first.return_value = transcription

        mock_element = MagicMock()
        mock_element.select.return_value.where.return_value.count.return_value = 1
        mock_element.type = "document"

        def mock_list_children(parent_id):
            q = MagicMock()
            if parent_id in ("d51ea3db-68ee-4cc0-a87f-736ee17c5f87",
                              "dfba9f5c-02de-478c-85c5-0ee780455433",
                              "cf29300f-40bf-4b61-be93-6cb631be8fab",
                              "fd5bee0a-83e8-4bdc-aa48-52331af2e151"):
                # Dossier racine → renvoie le document
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([document]))
            else:
                # Document → renvoie la page
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([page]))
            return q

        with patch("pipeline.extract_text.Element",       mock_element), \
             patch("pipeline.extract_text.Transcription",  mock_transcription_cls), \
             patch("pipeline.extract_text.list_children",  mock_list_children):
            from pipeline.extract_text import extract_to_txt
            extract_to_txt()

        # Vérifie qu'au moins un .txt a été créé quelque part dans data/text_files
        txt_files = list((tmp_path / "data" / "text_files").rglob("*.txt"))
        assert len(txt_files) > 0, "Aucun fichier .txt créé par extract_to_txt"

    def test_txt_content_matches_transcription(self, tmp_path, monkeypatch):
        """Le contenu du .txt correspond aux transcriptions des pages."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        content = "Nous défendons les valeurs républicaines."
        transcription = MagicMock()
        transcription.text = content

        page = MagicMock()
        page.id = "page_1"

        document = MagicMock()
        document.name = "EL136_doc_content"
        document.id   = "doc_id_content"

        mock_transcription_cls = MagicMock()
        mock_transcription_cls.select.return_value.where.return_value.first.return_value = transcription

        mock_element = MagicMock()
        mock_element.select.return_value.where.return_value.count.return_value = 1

        def mock_list_children(parent_id):
            q = MagicMock()
            if parent_id in ("d51ea3db-68ee-4cc0-a87f-736ee17c5f87",
                              "dfba9f5c-02de-478c-85c5-0ee780455433",
                              "cf29300f-40bf-4b61-be93-6cb631be8fab",
                              "fd5bee0a-83e8-4bdc-aa48-52331af2e151"):
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([document]))
            else:
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([page]))
            return q

        with patch("pipeline.extract_text.Element",       mock_element), \
             patch("pipeline.extract_text.Transcription",  mock_transcription_cls), \
             patch("pipeline.extract_text.list_children",  mock_list_children):
            from pipeline.extract_text import extract_to_txt
            extract_to_txt()

        txt_files = list((tmp_path / "data" / "text_files").rglob("*.txt"))
        assert len(txt_files) > 0
        for txt in txt_files:
            assert content in txt.read_text(encoding="utf-8")

    def test_no_txt_if_no_transcription(self, tmp_path, monkeypatch):
        """Aucun .txt créé si le document n'a pas de transcription."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()

        page = MagicMock()
        page.id = "page_empty"

        document = MagicMock()
        document.name = "EL136_empty_doc"
        document.id   = "doc_empty"

        mock_transcription_cls = MagicMock()
        mock_transcription_cls.select.return_value.where.return_value.first.return_value = None

        mock_element = MagicMock()
        mock_element.select.return_value.where.return_value.count.return_value = 1

        def mock_list_children(parent_id):
            q = MagicMock()
            if parent_id in ("d51ea3db-68ee-4cc0-a87f-736ee17c5f87",
                              "dfba9f5c-02de-478c-85c5-0ee780455433",
                              "cf29300f-40bf-4b61-be93-6cb631be8fab",
                              "fd5bee0a-83e8-4bdc-aa48-52331af2e151"):
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([document]))
            else:
                q.where.return_value.count.return_value = 1
                q.where.return_value.__iter__ = MagicMock(return_value=iter([page]))
            return q

        with patch("pipeline.extract_text.Element",       mock_element), \
             patch("pipeline.extract_text.Transcription",  mock_transcription_cls), \
             patch("pipeline.extract_text.list_children",  mock_list_children):
            from pipeline.extract_text import extract_to_txt
            extract_to_txt()

        txt_files = list((tmp_path / "data" / "text_files").rglob("*.txt"))
        assert len(txt_files) == 0, f"Des .txt ont été créés alors qu'il n'y a pas de transcription : {txt_files}"
