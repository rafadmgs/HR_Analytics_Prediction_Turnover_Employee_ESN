import pandas as pd


def create_satisfaction_moyenne(df):
    """
    Crée une variable représentant la satisfaction moyenne
    de l'employé sur les différents aspects du travail.
    """
    df = df.copy()

    colonnes_satisfaction = [
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso"
    ]

    df["satisfaction_moyenne"] = (
        df[colonnes_satisfaction].mean(axis=1)
    )

    return df


def create_evolution_evaluation(df):
    """
    Calcule l'évolution entre l'évaluation actuelle
    et l'évaluation précédente.
    """
    df = df.copy()

    df["evolution_evaluation"] = (
        df["note_evaluation_actuelle"]
        - df["note_evaluation_precedente"]
    )

    return df


def create_experience_avant_entreprise(df):
    """
    Calcule le nombre d'années d'expérience professionnelle
    acquises avant l'entrée dans l'entreprise.
    """
    df = df.copy()

    df["experience_avant_entreprise"] = (
        df["annee_experience_totale"]
        - df["annees_dans_l_entreprise"]
    )

    return df


def create_impact_deplacement(df):
    """
    Crée une variable indiquant l'impact potentiel
    de la fréquence des déplacements professionnels.
    """
    df = df.copy()

    df["impact_deplacement"] = (
        df["frequence_deplacement"]
        .map({
            "Aucun déplacement": 0,
            "Occasionnel": 1,
            "Fréquent": 2
        })
    )

    return df


def create_impact_augmentation_equilibre(df):
    """
    Crée une interaction entre l'augmentation salariale
    précédente et l'équilibre vie professionnelle / vie personnelle.
    """
    df = df.copy()

    df["impact_augmentation_equilibre"] = (
        df["augementation_salaire_precedente"]
        * df["satisfaction_employee_equilibre_pro_perso"]
    )

    return df


def create_e4_features(df):
    """
    Applique l'ensemble des features engineering créées
    dans le cadre de l'étape E4.
    """
    df = df.copy()

    df = create_satisfaction_moyenne(df)
    df = create_evolution_evaluation(df)
    df = create_experience_avant_entreprise(df)
    df = create_impact_deplacement(df)
    df = create_impact_augmentation_equilibre(df)

    return df