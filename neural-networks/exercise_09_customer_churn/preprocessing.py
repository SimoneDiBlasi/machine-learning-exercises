import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from config import PREPROCESSOR_PATH


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def create_preprocessor():
    """
    Create the preprocessing pipeline used by the neural network.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


def preprocess_data(
    preprocessor,
    X_train,
    X_validation,
    X_test
):
    """
    Fit the preprocessor on the training data
    and transform training, validation and test data.
    """

    X_train = preprocessor.fit_transform(X_train)

    X_validation = preprocessor.transform(
        X_validation
    )

    X_test = preprocessor.transform(
        X_test
    )

    return (
        X_train,
        X_validation,
        X_test
    )


def save_preprocessor(preprocessor):
    """
    Save the fitted preprocessor to disk.
    """

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )


def load_preprocessor():
    """
    Load the fitted preprocessor from disk.
    """

    return joblib.load(
        PREPROCESSOR_PATH
    )