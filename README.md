# HR Analytics – Employee Turnover Prediction

## Contexte

Une entreprise de services du numérique (ESN) souhaite mieux comprendre les facteurs associés au départ de ses employés.

L'entreprise dispose de plusieurs sources de données contenant des informations sur les employés, leur situation professionnelle, leurs évaluations et leur perception de leur environnement de travail.

L'objectif est de croiser et d'analyser ces données afin d'identifier les principaux facteurs associés à l'attrition et de préparer les données pour la modélisation prédictive.

## Objectif

Identifier les principaux facteurs associés à l'attrition des employés afin de mieux comprendre les causes de départ au sein de l'ESN et de préparer la construction d'un modèle prédictif.

## Objectifs Data

- Comprendre et nettoyer les différentes sources de données.
- Construire un DataFrame central à partir des fichiers disponibles.
- Réaliser une analyse exploratoire des variables qualitatives et quantitatives.
- Identifier les différences entre les employés ayant quitté l'entreprise et ceux restés.
- Identifier les variables potentiellement associées à l'attrition.
- Préparer les données pour la modélisation.
- Construire et évaluer des modèles de Machine Learning permettant de prédire l'attrition.

---

# Navigation

## ÉTAPE 1 – Analyse exploratoire et nettoyage

[Ouvrir le notebook E1](./notebooks/P4_E1_analyse_exploratoire.ipynb)

### Contenu

- Importation et exploration des sources de données
- Nettoyage des données
- Jointure des différentes sources
- Construction du DataFrame central
- Analyse des variables qualitatives
- Analyse des variables quantitatives
- Identification des facteurs associés à l'attrition
- Synthèse des principaux insights

---

## ÉTAPE 2 – Préparation des données pour la modélisation

[Ouvrir le notebook E2](./notebooks/P4_E2_preparation_modelisation.ipynb)

### Contenu

- Définition de la variable cible `y`
- Définition des variables explicatives `X`
- Identification des variables quantitatives et qualitatives
- Analyse des corrélations avec Pearson
- Analyse des relations non linéaires avec des pairplots
- Analyse complémentaire avec Spearman
- Sélection des variables fortement corrélées
- Sélection des méthodes d'encodage selon la nature des variables
- Encodage des variables qualitatives nominales avec `OneHotEncoder`
- Conservation numérique des variables ordinales
- Création d'un `ColumnTransformer`
- Vérification du preprocessing
- Création d'une fonction de préparation des données
- Validation et exportation du jeu de données final

### Données préparées

Le jeu de données final destiné à la modélisation contient :

- **1 470 observations**
- **46 features numériques**
- **1 variable cible** : `a_quitte_l_entreprise`
- **0 valeur manquante**

Les données préparées sont disponibles dans :

`data/processed/df_modelisation.csv`

---

# ÉTAPE 3 – Modélisation

> À venir

### Objectif

Construire et comparer plusieurs modèles de Machine Learning afin de prédire l'attrition des employés.

---

# ÉTAPE 4 – Évaluation et interprétation

> À venir

### Objectif

Évaluer les performances des modèles, identifier le modèle le plus pertinent et interpréter les principaux facteurs associés aux prédictions.

---

# Structure du projet

```text
HR_Analytics_Prediction_Turnover_Employee_ESN/
│
├── data/
│   ├── raw/
│   │   ├── extrait_sirh.csv
│   │   ├── extrait_eval.csv
│   │   └── extrait_sondage.csv
│   │
│   └── processed/
│       ├── df_central.csv
│       └── df_modelisation.csv
│
├── notebooks/
│   ├── P4_E1_analyse_exploratoire.ipynb
│   └── P4_E2_preparation_modelisation.ipynb
│
├── presentation/
│
├── src/
│   └── project_4/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
