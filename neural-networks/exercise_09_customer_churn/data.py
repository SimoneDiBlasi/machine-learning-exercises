import pandas as pd
from sklearn.model_selection import train_test_split
from config import (DATASET_PATH, RANDOM_STATE)


def load_dataset():
    """Load the Telco Customer Churn dataset."""
    return pd.read_csv(DATASET_PATH)


def clean_dataset(dataset):
    """Clean and prepare the dataset."""

    dataset["Churn"] = dataset["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    dataset.drop(columns=["customerID"], inplace=True)

    dataset["TotalCharges"] = pd.to_numeric(
        dataset["TotalCharges"], errors="coerce")

    dataset.dropna(inplace=True)

    return dataset


def split_dataset(dataset):
    """Split dataset into training, validation and test sets."""

    X = dataset.drop(columns=["Churn"])

    y = dataset["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )
