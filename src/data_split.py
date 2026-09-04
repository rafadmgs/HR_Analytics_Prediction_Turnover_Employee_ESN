from sklearn.model_selection import train_test_split


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
