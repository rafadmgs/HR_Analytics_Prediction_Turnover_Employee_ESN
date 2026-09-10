import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)


# ============================================================
# MODÈLES DE CLASSIFICATION
# ============================================================

def train_logistic_regression(
    X_train,
    y_train,
    random_state=42,
    max_iter=1000
):
    """
    Entraîne un modèle de régression logistique.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables explicatives du jeu d'apprentissage.

    y_train : pandas.Series
        Variable cible.

    random_state : int, default=42
        Graine aléatoire.

    max_iter : int, default=1000
        Nombre maximal d'itérations.

    Returns
    -------
    LogisticRegression
        Modèle de régression logistique entraîné.
    """

    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators=300,
    class_weight="balanced",
    random_state=42
):
    """
    Entraîne un modèle Random Forest avec pondération des classes.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables explicatives du jeu d'apprentissage.

    y_train : pandas.Series
        Variable cible.

    n_estimators : int, default=300
        Nombre d'arbres.

    class_weight : str or None, default="balanced"
        Pondération des classes.

    random_state : int, default=42
        Graine aléatoire.

    Returns
    -------
    RandomForestClassifier
        Modèle Random Forest entraîné.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# VALIDATION CROISÉE
# ============================================================

def cross_validate_random_forest(
    model,
    X,
    y,
    n_splits=5,
    shuffle=True,
    random_state=42
):
    """
    Réalise une validation croisée stratifiée du modèle Random Forest.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    scores = []

    for train_idx, val_idx in cv.split(X, y):

        X_train_cv = X.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]

        y_train_cv = y.iloc[train_idx]
        y_val_cv = y.iloc[val_idx]

        model_cv = clone(model)

        model_cv.fit(
            X_train_cv,
            y_train_cv
        )

        y_pred_cv = model_cv.predict(
            X_val_cv
        )

        score = f1_score(
            y_val_cv,
            y_pred_cv,
            pos_label="Oui"
        )

        scores.append(score)

    return np.array(scores)


# ============================================================
# COURBE PRECISION / RECALL
# ============================================================

def calculate_precision_recall_curve(
    model,
    X,
    y,
    positive_class="Oui"
):
    """
    Calcule les métriques Precision et Recall
    pour différents seuils de probabilité.
    """

    probabilities = model.predict_proba(
        X
    )

    positive_index = list(
        model.classes_
    ).index(positive_class)

    y_proba = probabilities[
        :,
        positive_index
    ]

    thresholds = np.linspace(
        0,
        1,
        101
    )

    precisions = []
    recalls = []

    for threshold in thresholds:

        y_pred = np.where(
            y_proba >= threshold,
            positive_class,
            "Non"
        )

        precisions.append(
            precision_score(
                y,
                y_pred,
                pos_label=positive_class,
                zero_division=0
            )
        )

        recalls.append(
            recall_score(
                y,
                y_pred,
                pos_label=positive_class,
                zero_division=0
            )
        )

    return (
        thresholds,
        np.array(precisions),
        np.array(recalls)
    )


# ============================================================
# PRÉDICTIONS OUT-OF-FOLD
# ============================================================

def generate_oof_predictions(
    X,
    y,
    n_splits=5,
    n_estimators=200,
    class_weight="balanced",
    random_state=42
):
    """
    Génère des prédictions out-of-fold pour l'ensemble des observations.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    oof_probabilities = np.zeros(
        len(X)
    )

    for train_idx, val_idx in cv.split(X, y):

        X_train_cv = X.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]

        y_train_cv = y.iloc[train_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(
            X_train_cv,
            y_train_cv
        )

        probabilities = model.predict_proba(
            X_val_cv
        )

        positive_index = list(
            model.classes_
        ).index("Oui")

        oof_probabilities[val_idx] = (
            probabilities[
                :,
                positive_index
            ]
        )

    return oof_probabilities


# ============================================================
# OPTIMISATION DU SEUIL
# ============================================================

def find_optimal_threshold(
    y_true,
    y_proba,
    positive_class="Oui"
):
    """
    Recherche le seuil maximisant le F1-score
    pour la classe positive.
    """

    thresholds = np.linspace(
        0.1,
        0.9,
        81
    )

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        y_pred = np.where(
            y_proba >= threshold,
            positive_class,
            "Non"
        )

        score = f1_score(
            y_true,
            y_pred,
            pos_label=positive_class,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# ÉVALUATION DES MODÈLES
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=0.5,
    positive_class="Oui"
):
    """
    Évalue un modèle de classification.

    Les métriques retournées sont :
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC

    Le seuil de décision peut être ajusté.
    """

    probabilities = model.predict_proba(
        X_test
    )

    positive_index = list(
        model.classes_
    ).index(positive_class)

    y_proba = probabilities[
        :,
        positive_index
    ]

    y_pred = np.where(
        y_proba >= threshold,
        positive_class,
        "Non"
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        (y_test == positive_class).astype(int),
        y_proba
    )

    return {
        "probabilities": y_proba,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

def calculate_confusion_matrix(
    y_true,
    y_pred,
    labels=("Non", "Oui")
):
    """
    Calcule la matrice de confusion.
    """

    from sklearn.metrics import confusion_matrix

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


# ============================================================
# RÉÉQUILIBRAGE DES CLASSES
# ============================================================

def apply_smote(
    X_train,
    y_train,
    random_state=42
):
    """
    Applique SMOTE afin de rééquilibrer les classes.
    """

    from imblearn.over_sampling import SMOTE

    smote = SMOTE(
        random_state=random_state
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    return X_resampled, y_resampled


# ============================================================
# RÉSULTATS DES PRÉDICTIONS
# ============================================================

def create_prediction_results(
    employee_ids,
    probabilities,
    predictions
):
    """
    Crée un DataFrame contenant les résultats des prédictions.
    """

    results = pd.DataFrame({
        "id_employee": employee_ids,
        "Probabilité d'attraction (%)": (
            probabilities * 100
        ),
        "Prédiction": predictions
    })

    return results


# ============================================================
# IMPORTANCE DES VARIABLES
# ============================================================

def get_feature_importance(
    model,
    feature_names
):
    """
    Retourne l'importance des variables
    d'un modèle Random Forest.
    """

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return importance


# ============================================================
# RANDOM FOREST - FINE-TUNING
# ============================================================

def create_random_forest(
    random_state=42
):
    """
    Crée une instance de Random Forest
    destinée au fine-tuning.
    """

    return RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )


def get_random_forest_param_distributions():
    """
    Définit les distributions de paramètres
    utilisées pour le RandomizedSearchCV.
    """

    return {
        "n_estimators": randint(
            200,
            800
        ),
        "max_depth": [
            None,
            5,
            10,
            15,
            20,
            30
        ],
        "min_samples_split": randint(
            2,
            20
        ),
        "min_samples_leaf": randint(
            1,
            10
        ),
        "max_features": [
            "sqrt",
            "log2",
            None
        ],
        "class_weight": [
            "balanced",
            "balanced_subsample"
        ]
    }


def create_randomized_search(
    model,
    param_distributions,
    n_iter=50,
    random_state=42
):
    """
    Crée le RandomizedSearchCV utilisé pour
    optimiser les hyperparamètres du modèle.

    L'optimisation est réalisée sur le F1-score
    de la classe 'Oui' avec une validation croisée
    stratifiée à 5 folds.
    """

    f1_oui = make_scorer(
        f1_score,
        pos_label="Oui"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=f1_oui,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    return random_search