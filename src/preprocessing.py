import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


def prepare_evaluation(df_evaluation):
    """
    Prépare le DataFrame d'évaluation pour les jointures.

    Transformations :
    - Conversion de 'augementation_salaire_precedente' de texte vers entier.
    - Harmonisation de 'eval_number' en supprimant le préfixe 'E_'.
    """
    df_evaluation = df_evaluation.copy()

    df_evaluation["augementation_salaire_precedente"] = (
        df_evaluation["augementation_salaire_precedente"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype("int64")
    )

    df_evaluation["eval_number"] = (
        df_evaluation["eval_number"]
        .str.replace("E_", "", regex=False)
        .astype("int64")
    )

    return df_evaluation


def remove_constant_columns(df_sirh, df_sondage):
    """
    Supprime les colonnes sans variabilité identifiées dans les données.
    """
    df_sirh = df_sirh.copy()
    df_sondage = df_sondage.copy()

    df_sirh = df_sirh.drop(
        columns=["nombre_heures_travailless"]
    )

    df_sondage = df_sondage.drop(
        columns=[
            "nombre_employee_sous_responsabilite",
            "ayant_enfants"
        ]
    )

    return df_sirh, df_sondage


def create_central_dataframe(
    df_sirh,
    df_evaluation,
    df_sondage
):
    """
    Crée le DataFrame central à partir des trois sources.

    Les jointures sont de type inner afin de conserver uniquement
    les employés présents dans les trois bases.
    """
    df_central = pd.merge(
        df_sirh,
        df_evaluation,
        left_on="id_employee",
        right_on="eval_number",
        how="inner"
    )

    df_central = pd.merge(
        df_central,
        df_sondage,
        left_on="id_employee",
        right_on="code_sondage",
        how="inner"
    )

    df_central = df_central.drop(
        columns=[
            "eval_number",
            "code_sondage"
        ]
    )

    return df_central


def prepare_modelisation_data(df):
    """
    Prépare les données pour la modélisation.

    Transformations :
    - séparation de la variable cible et des features ;
    - suppression de l'identifiant ;
    - suppression des variables fortement corrélées ;
    - encodage OneHot des variables qualitatives nominales ;
    - conservation des variables quantitatives et ordinales.

    Returns
    -------
    X : pandas.DataFrame
        Variables explicatives préparées pour la modélisation.

    y : pandas.Series
        Variable cible.
    """

    df = df.copy()

    # Variable cible
    y = df[
        "a_quitte_l_entreprise"
    ].copy()

    # Variables explicatives
    X = df.drop(
        columns=[
            "a_quitte_l_entreprise",
            "id_employee"
        ]
    ).copy()

    # Suppression des variables fortement corrélées
    X = X.drop(
        columns=[
            "annees_dans_le_poste_actuel",
            "annes_sous_responsable_actuel"
        ]
    )

    # Variables qualitatives nominales
    variables_nominales = [
        "genre",
        "statut_marital",
        "departement",
        "poste",
        "heure_supplementaires",
        "domaine_etude",
        "frequence_deplacement"
    ]

    # Création du préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                variables_nominales
            )
        ],
        remainder="passthrough"
    )

    # Transformation
    X_transformed = preprocessor.fit_transform(
        X
    )

    # Récupération des noms des variables
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Création du DataFrame final
    X = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )

    return X, y


def split_train_test(
    X,
    y,
    test_size=0.20,
    random_state=42
):
    """
    Sépare les données en jeux d'apprentissage et de test
    en conservant la proportion des classes.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train,
    X_test
):
    """
    Standardise les variables du jeu d'apprentissage
    et du jeu de test.

    Le scaler est ajusté uniquement sur le jeu
    d'apprentissage afin d'éviter toute fuite de données.

    Returns
    -------
    X_train_scaled : pandas.DataFrame
        Jeu d'apprentissage standardisé.

    X_test_scaled : pandas.DataFrame
        Jeu de test standardisé.
    """

    scaler = StandardScaler()

    # Ajustement uniquement sur le train
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # Transformation du test avec le scaler du train
    X_test_scaled = scaler.transform(
        X_test
    )

    # Conservation du format DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled


def create_satisfaction_moyenne(df):
    """
    Crée la satisfaction moyenne de l'employé
    à partir des quatre dimensions de satisfaction.
    """
    df = df.copy()

    colonnes_satisfaction = [
        "remainder__satisfaction_employee_environnement",
        "remainder__satisfaction_employee_nature_travail",
        "remainder__satisfaction_employee_equipe",
        "remainder__satisfaction_employee_equilibre_pro_perso"
    ]

    df["satisfaction_moyenne"] = (
        df[colonnes_satisfaction]
        .mean(axis=1)
    )

    return df


def create_evolution_evaluation(df):
    """
    Calcule l'évolution entre l'évaluation actuelle
    et l'évaluation précédente.
    """
    df = df.copy()

    df["evolution_evaluation"] = (
        df["remainder__note_evaluation_actuelle"]
        - df["remainder__note_evaluation_precedente"]
    )

    return df


def create_experience_avant_entreprise(df):
    """
    Calcule le nombre d'années d'expérience professionnelle
    acquises avant l'entrée dans l'entreprise.
    """
    df = df.copy()

    df["experience_avant_entreprise"] = (
        df["remainder__annee_experience_totale"]
        - df["remainder__annees_dans_l_entreprise"]
    )

    return df


def create_impact_deplacement(df):
    """
    Crée une variable représentant l'impact de la fréquence
    des déplacements professionnels pondérée par la distance
    domicile-travail.
    """
    df = df.copy()

    poids_deplacement = (
        df["nominal__frequence_deplacement_Occasionnel"]
        + 2
        * df["nominal__frequence_deplacement_Frequent"]
    )

    df["impact_deplacement"] = (
        df["remainder__distance_domicile_travail"]
        * poids_deplacement
    )

    return df


def create_impact_augmentation_equilibre(df):
    """
    Crée une interaction entre l'augmentation salariale
    précédente et l'équilibre vie professionnelle / vie personnelle.
    """
    df = df.copy()

    df["impact_augmentation_equilibre"] = (
        df["remainder__augementation_salaire_precedente"]
        * df[
            "remainder__satisfaction_employee_equilibre_pro_perso"
        ]
    )

    return df


def create_e4_features(df):
    """
    Applique l'ensemble des features engineering
    créées dans le cadre de l'étape E4.
    """
    df = df.copy()

    df = create_satisfaction_moyenne(
        df
    )

    df = create_evolution_evaluation(
        df
    )

    df = create_experience_avant_entreprise(
        df
    )

    df = create_impact_deplacement(
        df
    )

    df = create_impact_augmentation_equilibre(
        df
    )

    return df


def export_results(
    results,
    output_path
):
    """
    Exporte les résultats dans un fichier CSV.

    Parameters
    ----------
    results : pandas.DataFrame
        Tableau des résultats à exporter.

    output_path : str
        Chemin du fichier de sortie.
    """
    results.to_csv(
        output_path,
        index=False
    )

    print(
        "Résultats exportés avec succès."
    )


def create_prediction_results(
    employee_ids,
    probabilities,
    predictions
):
    """
    Crée un tableau récapitulatif des prédictions.

    Parameters
    ----------
    employee_ids : array-like
        Identifiants des employés.

    probabilities : array-like
        Probabilités prédites d'attraction.

    predictions : array-like
        Classes prédites.

    Returns
    -------
    pandas.DataFrame
        Tableau final des résultats.
    """

    results = pd.DataFrame({
        "id_employee": employee_ids,
        "Probabilité d'attraction (%)": (
            probabilities * 100
        ),
        "Prédiction": predictions
    })

    return results