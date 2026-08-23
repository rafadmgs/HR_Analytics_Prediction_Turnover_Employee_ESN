# HR Analytics – Employee Turnover Prediction

## Contexte

Une entreprise de services du numérique (ESN) souhaite mieux comprendre les facteurs associés au départ de ses employés.

L'entreprise dispose de plusieurs fichiers de données contenant des informations sur les employés, leur situation professionnelle et leurs caractéristiques.

L'objectif est d'explorer et de croiser ces données afin d'identifier les principales tendances associées à l'attrition et de préparer les données pour la modélisation prédictive.

## Objectif

Identifier les principaux facteurs associés à l'attrition des employés afin de mieux comprendre les causes de départ au sein de l'ESN et de préparer la construction d'un modèle prédictif.

## Objectifs Data

- Comprendre et nettoyer les différentes sources de données.
- Construire un DataFrame central à partir des fichiers disponibles.
- Réaliser une analyse exploratoire des variables qualitatives et quantitatives.
- Identifier les différences entre les employés ayant quitté l'entreprise et ceux restés.
- Faire ressortir les variables potentiellement associées à l'attrition.
- Préparer les données pour les étapes de modélisation.

---

# ÉTAPE 1 - Analyse exploratoire et Nettoyage

## Importation des librairies

Importation des principales bibliothèques Python utilisées pour l'analyse et la visualisation des données.

## Importation des jeux de données

Importation des trois sources de données :

- `df_sirh`
- `df_evaluation`
- `df_sondage`

## Analyse exploratoire

### Base `df_evaluation`

- Dimensions
- Statistiques descriptives
- Informations générales
- Doublons
- Valeurs manquantes
- Types des données
- Valeurs uniques
- Vérification des colonnes
- Identification des variables quantitatives et qualitatives

### Base `df_sirh`

- Dimensions
- Statistiques descriptives
- Informations générales
- Doublons
- Valeurs manquantes
- Types des données
- Valeurs uniques
- Vérification des colonnes
- Identification des variables quantitatives et qualitatives

### Base `df_sondage`

- Dimensions
- Statistiques descriptives
- Informations générales
- Doublons
- Valeurs manquantes
- Types des données
- Valeurs uniques
- Vérification des colonnes
- Identification des variables quantitatives et qualitatives

## Nettoyage des données

Nettoyage et préparation des différentes colonnes afin de rendre les données exploitables pour la jointure et l'analyse.

## Jointure des données

Identification des clés communes entre les trois sources et création d'un DataFrame centralisé.

### DataFrame central

Le `df_central` regroupe les informations issues des différentes sources et constitue la base utilisée pour l'analyse exploratoire.

## Analyse qualitative

Analyse de l'attrition selon différentes variables qualitatives :

- Heures supplémentaires
- Fréquence des déplacements
- Département
- Statut marital
- Genre
- Niveau d'éducation
- Poste
- Satisfaction envers l'environnement de travail
- Satisfaction envers la nature du travail
- Satisfaction envers l'équipe
- Équilibre vie professionnelle / vie personnelle

### Synthèse des variables qualitatives

Les variables sont classées selon leur relation probable avec l'attrition :

- **Forte**
- **Modérée**
- **Faible**

## Analyse quantitative

Analyse de la distribution des variables quantitatives selon l'attrition à l'aide notamment de boxplots.

Variables étudiées :

- Âge
- Revenu mensuel
- Expérience totale
- Ancienneté dans l'entreprise
- Ancienneté dans le poste actuel
- Nombre d'expériences précédentes
- Participation au PEE
- Nombre de formations suivies
- Distance domicile-travail
- Années depuis la dernière promotion
- Années sous le responsable actuel
- Augmentation salariale précédente

### Synthèse des variables quantitatives

Les variables sont classées selon leur relation probable avec l'attrition :

- **Forte**
- **Modérée**
- **Faible**

## Insights principaux

L'analyse exploratoire met notamment en évidence une attrition plus importante chez :

- Les employés effectuant des heures supplémentaires.
- Les employés effectuant des déplacements professionnels fréquents.
- Certains postes, notamment les Représentants Commercial.
- Les employés présentant une faible satisfaction, particulièrement concernant l'équilibre vie professionnelle / vie personnelle.
- Les employés ayant un revenu mensuel plus faible.
- Les employés ayant une ancienneté plus courte dans l'entreprise.
- Les employés ayant une distance domicile-travail plus importante.

Certaines variables, telles que le genre, les formations suivies et la participation au PEE, présentent peu de différences entre les employés ayant quitté l'entreprise et ceux restés.

> Ces résultats correspondent à des associations observées lors de l'analyse exploratoire et ne permettent pas de conclure à une relation de causalité.

---

## Structure du projet

```text
HR_Analytics_Prediction_Turnover_Employee_ESN/
│
├── data/
│   ├── extrait_sirh.csv
│   ├── extrait_eval.csv
│   └── extrait_sondage.csv
│
├── notebooks/
│   └── P4_E1_analyse_exploratoire.ipynb
│
├── README.md
└── .gitignore
