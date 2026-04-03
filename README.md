# Pipeline NLP — Détection de la langue de bois (1981–1993)

Analyse de la "langue de bois" dans les discours de campagne électorale française.

**Auteurs :**
- Ben Belgacem Dikra - @dikrabenbelgacem
- Tran Cindy - @cindyoff

ENSAE — MS Data Science — 2026

---

# Sommaire
1. [Introduction](#introduction)
2. [Données](#données)
3. [Structure du projet](#structure-du-projet)
4. [Architecture du projet](#architecture-du-projet)
5. [Démarche](#démarche)
6. [Lancement de la pipeline complète](#lancement-de-la-pipeline-complète)
7. [Conclusion](#conclusion)

# Introduction

La "langue de bois" désigne un style de discours politique caractérisé par des formules creuses et vagues ainsi que l'évitement du concret, un phénomène bien documenté dans la rhétorique électorale française, mais difficile à quantifier. 

Ce projet propose un pipeline NLP de bout en bout pour détecter la langue de bois au niveau de la phrase dans un corpus de 12 876 professions de foi et discours présidentiels couvrant les élections législatives et présidentielles françaises de 1981 à 1993 (source : base Arkindex / Sciences Po Archelec).

Faute de données annotées existantes, la démarche est semi-supervisée : des features linguistiques interprétables (concrétude, mots vagues, verbes modaux, entités nommées, sentiment) combinées à des embeddings contextuels CamemBERT alimentent deux modèles supervisés (régression logistique et XGBoost) entraînés sur un échantillon annoté manuellement d'environ 1 100 phrases. Les modèles sont ensuite appliqués à l'ensemble du corpus et les résultats sont explorables via deux dashboards sous Streamlit.

# Données

| Corpus | Documents | Type |
|---|---|---|
| 1981 / législatives | 3 182 | Profession de foi |
| 1988 / législatives | 3 628 | Profession de foi |
| 1988 / présidentielle | 130 | Discours présidentiel |
| 1993 / législatives | 5 936 | Profession de foi |
| **Total** | **12 876** | 369 478 phrases extraites |

Source : export SQLite Arkindex, Science Po Archelec corpus data

# Structure du projet

```
ml-nlp/
├── main.py                                         # Cindy Tran
├── dashboard.py                                    # Cindy Tran
├── statistique_resume.py                           # Cindy Tran
├── sample_annotation.py                            # Cindy Tran             
├── requirements.txt
├── annotate.py                                     # Cindy Tran
├── setup.py                                        # Cindy Tran
│
├── pipeline/
│   ├── __init__.py                                 # Cindy Tran  
│   ├── config.py                                   # Cindy Tran
│   ├── utils.py                                    # Cindy Tran
│   ├── extract_text.py                             # Cindy Tran
│   ├── sentences.py                                # Cindy Tran
│   ├── embedder.py                                 # Cindy Tran
│   ├── features_engineering.py                     # Cindy Tran
│   ├── merger.py                                   # Cindy Tran
│   ├── labeler.py                                  # Cindy Tran
│   └── modelisation.py                             # Cindy Tran
│
├── dictionnaire/
│   ├── dictionnaire_final_clean.txt                # Dikra Ben Belgacem
│   └── dictionnaire_langue_de_bois.txt             # Dikra Ben Belgacem
│
├── notebooks/
│   ├── analyse_langue_de_bois.ipynb                # Dikra Ben Belgacem
│
├── data/
│   ├── archelec_search.csv
│   ├── text_files/
│   │   ├── 1981/legislatives/*.txt
│   │   ├── 1988/legislatives/*.txt
│   │   ├── 1988/presidentielle/*.txt
│   │   └── 1993/legislatives/*.txt
│   ├── labels/
│   │   └── annotation_sample.csv                    # Cindy Tran
│   └── lexicons/
│       ├── dictionnaire_final_clean.txt             # Dikra Ben Belgacem
│       └── modal_verbs.txt                          # Cindy Tran
│
├── tests/
│   ├── __init__.py                                  # Cindy Tran
│   ├── test_extract.py                              # Cindy Tran
│   ├── test_labeler.py                              # Cindy Tran
│   ├── test_merger.py                               # Cindy Tran
│   ├── test_modelisation.py                         # Cindy Tran
│   └── test_sentences.py                            # Cindy Tran
│
└── outputs/
    ├── sentences.parquet            
    ├── embeddings.parquet
    ├── features.parquet            
    ├── final.parquet         
    ├── final_labeled.parquet      
    ├── final_labeled_2class.parquet 
    ├── final_predicted.parquet      
    └── models/
        ├── lr_estimator.joblib     
        ├── xgb_estimator.joblib    
        ├── lr_params.json           
        ├── xgb_params.json         
        └── evaluation.json          
```

# Architecture du projet

- `sentences.py` : nettoyage des textes et segmentation avec NLTK. Avant la tokenisation, un filtre administratif supprime ligne par ligne les en-têtes et pieds de page (références "Sciences Po / CEVIPOF", imprimeries, noms de candidats seuls, lignes biographiques, lignes courtes sans verbe conjugué). Les phrases de moins de 4 mots issues de la tokenisation sont également écartées.
  - Ce **`filter_ratio`** est stocké comme colonne dans `sentences.parquet` et se propage automatiquement jusqu'à `final.parquet`. C'est une feature documentaire : elle donne au modèle une information sur le contexte de bruit de chaque phrase.

- `embedder.py` : encode chaque phrase avec CamemBERT via un mean pooling sur les tokens. Usage automatique de MPS, CUDA ou CPU. Les embeddings sont ensuite stockés en colonne

- `features_engineering.py` : calcul de 5 catégories de features linguistiques indépendamment des embeddings :
  - Concrétude
  - Mots vagues
  - Verbes modaux
  - Entités nommées
  - Sentiment
  - Bruit documentaire

- `merger.py` : jointure entre `embeddings.parquet` et `features.parquet` sur `PRIMARY_KEY` pour produire `final.parquet`

- `labeler.py`
Joint le CSV d'annotation sur `PRIMARY_KEY` et ajoute la colonne `label` au parquet. Les phrases sans entrée dans le CSV reçoivent `NaN`

- `modelisation.py` : entraînement de deux modèles supervisés sur les phrases annotées (split stratifié 70 test / 15 validation / 15 train) :
  - Régression logistique : features linguistiques uniquement
    - Sélection par Information Value (IV)
    - Test de significativité de Wald
    - Calibration du seuil de Bayes optimal sur le set de validation
  - XGBoost : features linguistiques + embeddings CamemBERT
    - Optimisation des hyperparamètres par Random Grid Search 
    - Calibration du seuil de Bayes optimal sur la validation

Les deux modèles prédisent ensuite l'ensemble du corpus (~110 000 phrases) et produisent `final_predicted.parquet` avec quatre colonnes ajoutées : `proba_lr`, `pred_lr`, `proba_xgb`, `pred_xgb`.

## Dashboards

- `dashboard.py` : visualisation par document
  - Affichage du texte de chaque document avec les phrases détectées comme langue de bois surlignées. L'intensité de la couleur est proportionnelle à la probabilité prédite. Un sélecteur permet de basculer entre LR et XGBoost, de filtrer par année / type d'élection, et d'ajuster le seuil d'affichage.

- `statistique_resume.py` : résumé statistique des modèles ; dashboard en deux colonnes et section résidus
  - Gauche (performances) : tableaux AUC, F1, précision, recall, accuracy par split avec histogrammme comparatif pour les deux modèles
  - Droite (propriétés) : features sélectionnées avec IV et p-value Wald (régression logistique), hyperparamètres random grid search et AUC (XGBoost)
  - Résidus : 
    - Test de Hosmer-Lemeshow (chi-square, p-value, histogramme observés vs attendus par décile)
    - Courbe de calibration
    - Résidus de Pearson
    - Résidus de déviance vs probabilité prédite
    - Distribution des probabilités par classe réelle

# Démarche

## Approche

- Représentation des phrases avec deux types de features complémentaires :
  - Features linguistiques interprétables (concrétude, mots vagues, verbes modaux, entités nommées, sentiment) construites à partir de lexiques et de règles, ancrées dans la littérature sur la rhétorique en politique
  - Embeddings CamemBERT (768 dims, mean pooling) pour capturer le sens contextuel

- Annotation manuelle ciblée sur un sous-échantillon de ~1 100 phrases tiré de façon stratifiée sur des documents complets. Ces annotations constituent le fondement pour entraîner et évaluer les modèles de classification

- Classification supervisée puis prédiction sur l'ensemble du corpus

- Visualisation : deux dashboards Streamlit pour explorer les prédictions et évaluer statistiquement les modèles

## Choix techniques

- Filtre administratif en cascade : les en-têtes (nom du candidat, imprimerie, "Sciences Po / CEVIPOF") sont détectés par une série de règles (marqueurs explicites, patterns biographiques, absence de verbe conjugué)
- `filter_ratio` comme feature : calcul de la proportion de lignes détectées comme administratives dans chaque document. Ce ratio est stocké dans `sentences.parquet` et propagé  jusqu'à `final.parquet`. Un document avec 70% de lignes "admin" est structurellement différent d'un discours homogène, cette information aide le modèle à pondérer les phrases
- Limite de 110 000 phrases : volume suffisant pour des statistiques descriptives représentatives, sans exploser les temps de calcul
- Fichier CSV comme source de vérité pour les labels : format simple, versionnable, indépendant de l'ordre de tokenisation. Ensuite, jointure sur `PRIMARY_KEY`
- Double sélection de variables (régression logistique) : Information Value (IV ≥ 0.02) puis test de Wald (p ≤ 0.05) pour ne conserver que les features statistiquement pertinentes
- Seuil de Bayes calibré : pour les deux modèles, le seuil de décision est optimisé sur le jeu de validation en minimisant le risque de Bayes

# Lancement de la pipeline complète

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
source venv/bin/activate

# Installation des packages requis
pip install -r requirements.txt

# spaCy français
python -m spacy download fr_core_news_md

# Annotation
python annotate.py

# Pipeline complète
python main.py --steps sentences embeddings features_engineering merger label modelise statistique_resume

# Lancement du second dashboard
streamlit run dashboard.py
```

# Conclusion

Ce projet présente une manière de détection de la langue de bois dans des discours électoraux historiques, en l'absence de données annotées préexistantes. La combinaison de features linguistiques interprétables et d'embeddings CamemBERT permet aux deux modèles de capturer à la fois les signaux lexicaux de surface (mots vagues, verbes modaux) et le sens contextuel des phrases.

La contrainte d'annotation manuelle reste le principal facteur limitant : un corpus annoté plus large améliorerait mécaniquement les performances et permettrait d'évaluer la robustesse inter-élections. Par ailleurs, la comparaison entre le mode filtré (2 classes) et le mode trois classes ouvre une piste pour mieux mesurer l'impact du bruit documentaire sur la classification.