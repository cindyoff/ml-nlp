# Pipeline NLP — Détection de la langue de bois (1981–1993)

Analyse automatique de la "langue de bois" dans les discours de campagne électorale française.
La pipeline extrait les transcriptions d'une base Arkindex, segmente les textes en phrases,
calcule des embeddings CamemBERT et des features linguistiques, entraîne des classifieurs
supervisés, puis visualise les résultats via deux dashboards interactifs.

**Auteurs :**
- Ben Belgacem Dikra (dikrabenbelgacem)
- Tran Cindy (cindyoff)

ENSAE — MS Data Science — 2026

---

## Données

| Corpus | Documents | Type |
|---|---|---|
| 1981 / législatives | 3 182 | Profession de foi |
| 1988 / législatives | 3 628 | Profession de foi |
| 1988 / présidentielle | 130 | Discours présidentiel |
| 1993 / législatives | 5 936 | Profession de foi |
| **Total** | **12 876** | 369 478 phrases extraites |

Source : export SQLite Arkindex (`data/sciencespo-archelec-20260217-121320.sqlite`)

---

## Structure du projet

```
ml-nlp/
├── main.py                          # Orchestrateur de la pipeline
├── dashboard.py                     # Visualisation des prédictions par document
├── statistique_resume.py            # Dashboard résumé statistique des modèles
├── sample_annotation.py             # Génération du CSV d'annotation (one-shot)
├── requirements.txt
│
├── pipeline/
│   ├── config.py                    # Paramètres globaux (modèles, chemins, lexiques)
│   ├── utils.py                     # Chargement des lexiques
│   ├── extract_text.py              # Extraction SQLite → fichiers .txt
│   ├── sentences.py                 # Segmentation en phrases → sentences.parquet
│   ├── embedder.py                  # Embeddings CamemBERT → embeddings.parquet
│   ├── features_engineering.py      # Features linguistiques → features.parquet
│   ├── merger.py                    # Fusion embeddings + features → final.parquet
│   ├── labeler.py                   # Application des labels CSV → final_labeled.parquet
│   └── modelisation.py              # Entraînement LR + XGBoost → outputs/models/
│
├── data/
│   ├── sciencespo-archelec-*.sqlite # Base Arkindex (12 Go)
│   ├── text_files/                  # Transcriptions extraites
│   │   ├── 1981/legislatives/*.txt
│   │   ├── 1988/legislatives/*.txt
│   │   ├── 1988/presidentielle/*.txt
│   │   └── 1993/legislatives/*.txt
│   ├── labels/
│   │   └── annotation_sample.csv    # ~1 100 phrases annotées manuellement (source de vérité)
│   └── lexicons/
│       ├── vague_words.txt          # Mots vagues abstraits (31 termes)
│       ├── modal_verbs.txt          # Verbes modaux français (24 entrées)
│       └── langue_de_bois.txt       # Lexique complémentaire (à enrichir)
│
└── outputs/
    ├── sentences.parquet            # Phrases segmentées (≤ 110 000)
    ├── embeddings.parquet           # Vecteurs CamemBERT (768 dims, float32)
    ├── features.parquet             # Features linguistiques (17 colonnes)
    ├── final.parquet                # Fusion embeddings + features
    ├── final_labeled.parquet        # Données avec colonne "label"
    ├── final_labeled_2class.parquet # Idem, restreint aux 2 classes annotées
    ├── final_predicted.parquet      # Corpus complet avec prédictions LR + XGBoost
    └── models/
        ├── lr_estimator.joblib      # Pipeline sklearn (StandardScaler + LogisticRegression)
        ├── xgb_estimator.joblib     # Modèle XGBClassifier (meilleurs hyperparamètres)
        ├── lr_params.json           # Features sélectionnées, IV, p-values, seuil
        ├── xgb_params.json          # Hyperparamètres, AUC CV, seuil
        └── evaluation.json          # Métriques train / val / test (LR + XGBoost)
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Modèle spaCy français
python -m spacy download fr_core_news_md

# Données NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Démarche

### Objectif

Détecter automatiquement la "langue de bois" dans des discours de campagne électorale française
(1981–1993). La langue de bois désigne un style rhétorique caractérisé par l'usage de formules
vagues, de verbes modaux, d'abstractions et d'un faible ancrage factuel.

### Approche

Faute de données annotées existantes, la démarche est semi-supervisée :

1. **Représentation des phrases** — deux types de features complémentaires :
   - *Features linguistiques interprétables* (concrétude, mots vagues, verbes modaux, entités nommées, sentiment) construites à partir de lexiques et de règles, ancrées dans la littérature sur la rhétorique politique
   - *Embeddings CamemBERT* (768 dims, mean pooling) pour capturer le sens contextuel

2. **Annotation manuelle ciblée** — plutôt que d'annoter les 369 478 phrases, un échantillon
   de ~1 100 phrases est tiré de façon **stratifiée** (proportionnel à chaque année × type
   d'élection) sur des **documents complets** (pas de phrases orphelines). Ces annotations
   constituent la vérité terrain pour entraîner et évaluer un classifieur.

3. **Classification supervisée** — entraînement sur les ~1 100 phrases annotées avec deux modèles
   complémentaires, puis prédiction sur l'ensemble du corpus (~110 000 phrases).

4. **Visualisation** — deux dashboards Streamlit pour explorer les prédictions et évaluer
   statistiquement les modèles.

### Deux modes de prétraitement comparables

La pipeline peut être exécutée dans deux modes configurés dans `pipeline/config.py` sous `PROCESSING_MODES` :

| Mode | `--mode` | Classes | Comportement |
|---|---|---|---|
| **Filtré** (défaut) | `filtered` | 2 | Lignes admin supprimées avant tokenisation |
| **Trois classes** | `three_class` | 3 | Rien supprimé — les phrases admin annotées `autre` |

Les outputs sont séparés par un suffixe (`_3class`) pour permettre la comparaison côte à côte :

```
outputs/sentences.parquet          ←  mode filtered
outputs/sentences_3class.parquet   ←  mode three_class
outputs/final_labeled.parquet
outputs/final_3class_labeled.parquet
```

L'objectif est de comparer les deux classifieurs en fin de projet : le mode `filtered` ignore le bruit en amont, le mode `three_class` apprend à le reconnaître comme une classe à part entière.

### Choix techniques notables

- **Filtre administratif en cascade** : les en-têtes (nom du candidat, imprimerie, "Sciences Po / CEVIPOF")
  sont détectés par une cascade de règles (marqueurs explicites, patterns biographiques, absence de verbe
  conjugué). En mode `filtered` ces lignes sont supprimées ; en mode `three_class` elles sont conservées
  pour être annotées manuellement comme `autre`.
- **`filter_ratio` comme feature** : dans les deux modes, on calcule la proportion de lignes détectées
  comme administratives dans chaque document. Ce ratio est stocké dans `sentences.parquet` et propagé
  jusqu'à `final.parquet`. Un document avec 70% de lignes "admin" est structurellement différent d'un
  discours homogène — cette information aide le modèle à pondérer les phrases.
- **Limite de 110 000 phrases** : volume suffisant pour des statistiques descriptives représentatives,
  sans exploser les temps de calcul (~30–40 min pour features + embeddings).
- **CSV comme source de vérité** pour les labels : format simple, versionnable, indépendant de l'ordre
  de tokenisation — jointure sur `PRIMARY_KEY`. Le labeler valide les valeurs selon le mode.
- **Double sélection de variables (LR)** : Information Value (IV ≥ 0.02) puis test de Wald (p ≤ 0.05)
  pour ne conserver que les features statistiquement pertinentes.
- **Seuil de Bayes calibré** : pour les deux modèles, le seuil de décision est optimisé sur le jeu de
  validation en minimisant le risque de Bayes (coûts FP/FN égaux → maximise le F1).

---

## Pipeline

### Vue d'ensemble

```
SQLite (Arkindex)
      │
      ▼
 extract_text       →  data/text_files/{année}/{type}/*.txt
      │
      ▼
  sentences              outputs/sentences.parquet
(NLTK + filtre admin)    (doc_id, PRIMARY_KEY, date, classe, sentence, filter_ratio)
  max 110 000 phrases    Documents complets, lignes admin filtrées
      │
      ├──────────────────────────────┐
      ▼                              ▼
  embedder                 features_engineering
(CamemBERT, mean pooling) (spaCy NER + sentiment + règles)
      │                              │
      ▼                              ▼
embeddings.parquet           features.parquet
(PRIMARY_KEY + embedding[768])  (17 features linguistiques)
      │                              │
      └──────────────┬───────────────┘
                     ▼
                  merger           →  outputs/final.parquet
                     │
                     ▼
                  labeler          ←  data/labels/annotation_sample.csv
                     │
                     ▼
          outputs/final_labeled.parquet
          (colonne "label" sur ~1 100 phrases annotées)
                     │
                     ▼
               modelisation
          (LR + XGBoost, split 70/15/15)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  outputs/models/        outputs/final_predicted.parquet
  (estimateurs +          (proba_lr, pred_lr,
   params + métriques)     proba_xgb, pred_xgb)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      dashboard         statistique_resume
  (visualisation        (performances + résidus
   par document)         des modèles)
```

> `embeddings` et `features_engineering` sont **indépendantes** — elles lisent toutes
> les deux `sentences.parquet` sans dépendance entre elles.

### Lancer la pipeline complète

```bash
# Mode filtré (défaut, 2 classes)
python main.py

# Mode trois classes (rien filtré, annoter "autre" manuellement)
python main.py --mode three_class
```

### Lancer des étapes spécifiques

```bash
python main.py --steps sentences embeddings features_engineering merger label modelise
```

Étapes disponibles (dans l'ordre) :
`index` → `open` → `extract` → `sentences` → `embeddings` → `features_engineering` → `merger` → `label` → `modelise` → `dashboard` → `statistique_resume`

---

## Modules en détail

### `extract_text.py`
Indexe la base SQLite puis extrait les transcriptions par dossier Arkindex (identifiés par UUID).
Chaque document devient un fichier `.txt` dans `data/text_files/{année}/{type}/`.

### `sentences.py`
Nettoie les textes et segmente avec NLTK (tokeniseur français). Avant la tokenisation,
un **filtre administratif** supprime ligne par ligne les en-têtes et pieds de page
(références "Sciences Po / CEVIPOF", imprimeries, noms de candidats seuls, lignes
biographiques, lignes courtes sans verbe conjugué). Les phrases de moins de 4 mots
issues de la tokenisation sont également écartées.

À l'exécution, le terminal affiche pour chaque groupe (année × type) le **ratio de
filtrage moyen** (proportion de lignes supprimées par document) et un résumé global :

```
  [1981/legislatives] 3182 fichier(s) — ratio filtré moy 38.4%  (min 12.0% / max 91.2%)
  [1988/legislatives] 3628 fichier(s) — ratio filtré moy 41.1%  (min 9.3% / max 95.0%)
  ...
✅ 110000 phrases extraites depuis 8240 documents
   Ratio de filtrage moyen : 39.8%  (1 240 documents avec > 50% de lignes filtrées)
```

Ce **`filter_ratio`** est stocké comme colonne dans `sentences.parquet` et se propage
automatiquement jusqu'à `final.parquet`. C'est une feature documentaire : elle donne
au modèle une information sur le contexte de bruit de chaque phrase.

Le paramètre `--max_sentences` limite le corpus à des documents complets jusqu'à atteindre le seuil.

```bash
python -m pipeline.sentences \
    --data_dir      data/text_files/ \
    --output        outputs/sentences.parquet \
    --max_sentences 110000
```

### `embedder.py`
Encode chaque phrase avec **CamemBERT** (`camembert-base`) via un mean pooling sur les tokens
non-padding (meilleur que le token `[CLS]` pour les tâches sentence-level). Utilise
automatiquement MPS (Apple Silicon), CUDA ou CPU. Les embeddings sont stockés en colonne
array `float32` (768 dimensions, compression snappy).

```bash
python -m pipeline.embedder \
    --input      outputs/sentences.parquet \
    --output     outputs/embeddings.parquet \
    --model      camembert-base \
    --batch_size 256
```

### `features_engineering.py`
Calcule 5 catégories de features linguistiques indépendamment des embeddings :

| Catégorie | Colonnes | Outil |
|---|---|---|
| Concrétude | `n_digits`, `n_dates`, `n_money`, `n_percent`, `numeric_ratio` | Regex |
| Mots vagues | `n_vague_words`, `vague_ratio` | Lexique `vague_words.txt` |
| Verbes modaux | `n_modal_verbs`, `modal_ratio` | Lexique `modal_verbs.txt` |
| Entités nommées | `n_ent_org`, `n_ent_loc`, `n_ent_law`, `n_ent_total` | spaCy NER |
| Sentiment | `sentiment_positive`, `sentiment_negative`, `sentiment_intensity` | distilcamembert |
| Bruit documentaire | `filter_ratio` | hérité de `sentences.parquet` |

Optimisations : spaCy en mode NER uniquement, modèle sentiment sur MPS, features
réglementaires en multiprocessing (≈ 3h → 30–40 min).

```bash
python -m pipeline.features_engineering \
    --input  outputs/sentences.parquet \
    --output outputs/features.parquet
```

### `merger.py`
Joint `embeddings.parquet` et `features.parquet` sur `PRIMARY_KEY` pour produire
`final.parquet` — une seule source de vérité avec toutes les colonnes.

```bash
python -m pipeline.merger \
    --embeddings outputs/embeddings.parquet \
    --features   outputs/features.parquet \
    --output     outputs/final.parquet
```

### `labeler.py`
Joint le CSV d'annotation sur `PRIMARY_KEY` et ajoute la colonne `label` au parquet.
Les phrases sans entrée dans le CSV reçoivent `NaN`.

```bash
python -m pipeline.labeler \
    --labels_csv data/labels/annotation_sample.csv \
    --input      outputs/final.parquet \
    --output     outputs/final_labeled.parquet
```

### `modelisation.py`
Entraîne deux classifieurs sur les phrases annotées (split stratifié 70 / 15 / 15) :

**Régression logistique** — features linguistiques uniquement :
1. Sélection par Information Value (IV ≥ 0.02)
2. Test de significativité de Wald (p ≤ 0.05) via statsmodels
3. Calibration du seuil de Bayes optimal sur la validation

**XGBoost** — features linguistiques + embeddings CamemBERT (785 dims) :
1. Optimisation des hyperparamètres par `RandomizedSearchCV` (CV stratifié, scoring AUC)
2. Calibration du seuil de Bayes optimal sur la validation

Les deux modèles prédisent ensuite l'ensemble du corpus (~110 000 phrases) et produisent
`final_predicted.parquet` avec quatre colonnes ajoutées : `proba_lr`, `pred_lr`, `proba_xgb`, `pred_xgb`.

```bash
python -m pipeline.modelisation \
    --input   outputs/final_labeled.parquet \
    --full    outputs/final.parquet \
    --output  outputs/models/
```

Options disponibles :

| Option | Défaut | Description |
|---|---|---|
| `--test_size` | `0.15` | Proportion du jeu de test |
| `--val_size` | `0.15` | Proportion du jeu de validation |
| `--n_iter` | `30` | Itérations RandomizedSearchCV XGBoost |
| `--seed` | `42` | Graine aléatoire |

---

## Dashboards

### `dashboard.py` — Visualisation par document

Affiche le texte de chaque document avec les phrases détectées comme langue de bois
surlignées. L'intensité de la couleur est proportionnelle à la probabilité prédite.
Un sélecteur permet de basculer entre LR et XGBoost, de filtrer par année / type
d'élection, et d'ajuster le seuil d'affichage.

```bash
streamlit run dashboard.py
# ou via main.py :
python main.py --steps dashboard
```

### `statistique_resume.py` — Résumé statistique des modèles

Dashboard en deux colonnes + section résidus :

- **Gauche — Performances** : tableaux AUC / F1 / Précision / Rappel / Accuracy par split
  (train / val / test) avec bar chart comparatif, pour LR et XGBoost.
- **Droite — Propriétés** : features sélectionnées avec IV et p-value Wald (LR),
  hyperparamètres RandomizedSearch et AUC CV (XGBoost).
- **Résidus** (sélecteur test / val / train, onglets LR / XGBoost) :
  - Test de Hosmer-Lemeshow (χ², p-value, histogramme observés vs attendus par décile)
  - Courbe de calibration (reliability diagram)
  - Résidus de Pearson (histogramme + test Shapiro-Wilk)
  - Résidus de déviance vs probabilité prédite
  - Distribution des probabilités par classe réelle

```bash
streamlit run statistique_resume.py
# ou via main.py :
python main.py --steps statistique_resume
```

---

## Annotation manuelle

### 1. Générer le CSV d'annotation (une seule fois)

```bash
python sample_annotation.py
```

Produit `data/labels/annotation_sample.csv` (~1 100 phrases, documents complets,
tirage stratifié par année × type d'élection, reproductible via `--seed`).

### 2. Annoter

Ouvrir `annotation_sample.csv` (Excel, Numbers, VSCode…) et remplir la colonne `label` :

| PRIMARY_KEY | doc_id | date | classe | sentence | label |
|---|---|---|---|---|---|
| EL136_..._4 | EL136_... | 1981 | legislatives | Nous allons construire... | `langue_de_bois` |
| EL136_..._5 | EL136_... | 1981 | legislatives | Le budget alloué est... | `non_langue_de_bois` |

Les lignes avec `label` vide sont ignorées — l'annotation peut être partielle et progressive.

### 3. Appliquer les labels

```bash
python main.py --steps label
```

---

## Schéma des parquets

### `sentences.parquet`
| Colonne | Type | Description |
|---|---|---|
| `doc_id` | str | Nom du fichier source |
| `PRIMARY_KEY` | str | `{doc_id}_{index}` — clé unique par phrase |
| `date` | str | Année (`1981`, `1988`, `1993`) |
| `classe` | str | Type d'élection (`legislatives`, `presidentielle`) |
| `sentence` | str | Texte de la phrase |
| `filter_ratio` | float | Proportion de lignes filtrées dans le document source (0 = aucune, 1 = tout) |

### `embeddings.parquet`
| Colonne | Type | Description |
|---|---|---|
| `PRIMARY_KEY` | str | Clé de jointure |
| `doc_id` | str | Document source |
| `date` | str | Année |
| `classe` | str | Type d'élection |
| `embedding` | list[float32] | Vecteur CamemBERT (768 dims) |

### `features.parquet`
Toutes les colonnes de `sentences.parquet` + les 17 features linguistiques.

### `final.parquet`
Toutes les colonnes de `features.parquet` + colonne `embedding`.

### `final_labeled.parquet`
Toutes les colonnes de `final.parquet` + colonne `label` (`NaN` si non annoté).

### `final_predicted.parquet`
Toutes les colonnes de `final.parquet` + quatre colonnes de prédiction :

| Colonne | Type | Description |
|---|---|---|
| `proba_lr` | float | Probabilité langue_de_bois selon la régression logistique |
| `pred_lr` | str | Label prédit par LR (seuil de Bayes calibré) |
| `proba_xgb` | float | Probabilité langue_de_bois selon XGBoost |
| `pred_xgb` | str | Label prédit par XGBoost (seuil de Bayes calibré) |

---

## Configuration (`pipeline/config.py`)

| Paramètre | Valeur | Description |
|---|---|---|
| `BERT_MODEL` | `camembert-base` | Modèle d'embedding |
| `BATCH_SIZE` | `256` | Taille des batchs d'encodage |
| `MAX_LENGTH` | `128` | Longueur max en tokens |
| `SPACY_MODEL` | `fr_core_news_md` | Modèle spaCy pour le NER |
| `SENTIMENT_MODEL` | `cmarkea/distilcamembert-base-sentiment` | Modèle de sentiment |
| `VAGUE_WORDS` | `vague_words.txt` + `langue_de_bois.txt` | Lexique mots vagues |
| `MODAL_VERBS` | `modal_verbs.txt` | Lexique verbes modaux |
