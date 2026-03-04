# Plan de travail — Analyse de la langue de bois dans le corpus Archelec

## Deadline : 30 avril 2026

---

## Dikra — Construction du dictionnaire

- [ ] Lire manuellement une dizaine de professions de foi (URLs OCR dans le CSV) et noter les expressions langue de bois repérées à l'œil
- [ ] Créer le fichier `dictionnaire/dictionnaire_langue_de_bois.txt` avec une expression/mot par ligne à partir des lectures
- [ ] Télécharger la base Brysbaert, filtrer les mots très abstraits (score < 2) et les traduire en français avec Helsinki-NLP
- [ ] Fusionner les deux sources pour obtenir le dictionnaire final

## Cindy — Scoring et analyse

- [ ] Récupérer et nettoyer les textes OCR via les URLs du CSV
- [ ] Calculer le score de langue de bois pour chaque document (ratio mots du dictionnaire / total des mots)
- [ ] Faire une analyse descriptive du corpus (distribution par année, parti, profession, longueur des textes)
- [ ] Croiser les scores par année (1981, 1988, 1993), par parti politique et par profession du candidat
- [ ] Produire des visualisations (boxplots, courbes temporelles)
- [ ] Interpréter les résultats et rédiger la discussion

## Commun — Rapport NeurIPS

- [ ] Introduction + problématique
- [ ] État de l'art (langue de bois, NLP politique, readability)
- [ ] Description des données et analyse descriptive
- [ ] Méthodologie (construction du dictionnaire, scoring)
- [ ] Résultats et visualisations
- [ ] Discussion et conclusion
- [ ] Dépôt du code + PDF sur GitHub avant le 30 avril