# HR Analytics – Employee Turnover Prediction

## Project Overview

This project aims to analyze employee turnover within an ESN (Entreprise de Services du Numérique).

The objective is to identify factors potentially associated with employee attrition and, in later stages, develop predictive models.

## E1 – Exploratory Data Analysis

The first stage focuses on understanding and exploring the HR data.

### Completed

- Analysis of the three original datasets
- Data cleaning and preparation
- Identification of qualitative and quantitative variables
- Creation of a centralized DataFrame through data integration
- Descriptive statistics
- Exploratory visualizations
- Analysis of attrition according to qualitative and quantitative variables
- Synthesis of the main exploratory insights

### Main exploratory insights

The analysis suggests that attrition is particularly associated with:

- Overtime
- Frequent business travel
- Certain job positions
- Lower satisfaction, especially regarding work-life balance
- Lower monthly income
- Lower company tenure
- Greater distance between home and workplace

Some variables, such as gender, training and PEE participation, show little difference between employees who left and those who remained.

These observations represent associations identified during exploratory analysis and do not imply causality.

# HR Analytics – Employee Turnover Prediction

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
