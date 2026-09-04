import pandas as pd


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


def create_central_dataframe(df_sirh, df_evaluation, df_sondage):
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
        columns=["eval_number", "code_sondage"]
    )

    return df_central