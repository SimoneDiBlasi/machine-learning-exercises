import pandas as pd
from pathlib import Path
from prediction import (
    load_model,
    load_preprocessor,
    load_threshold,
    predict_customer
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

EXERCISE_DIR = Path(__file__).resolve().parent

NEURAL_NETWORKS_DIR = EXERCISE_DIR.parent

MODEL_DIR = (
    NEURAL_NETWORKS_DIR
    / "exercise_09_customer_churn"
    / "models"
)


MODEL_PATH = (
    MODEL_DIR
    / "customer_churn_model.keras"
)

PREPROCESSOR_PATH = (
    MODEL_DIR
    / "customer_churn_preprocessor.joblib"
)

CONFIG_PATH = (
    MODEL_DIR
    / "customer_churn_config.json"
)


def main():

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load trained model
    # -----------------------------------------------------

    print("\n[1/4] Loading trained model...")

    model = load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # -----------------------------------------------------
    # 2. Load preprocessing pipeline
    # -----------------------------------------------------

    print(
        "\n[2/4] Loading preprocessing pipeline..."
    )

    preprocessor = load_preprocessor(
        PREPROCESSOR_PATH
    )

    print(
        "Preprocessor loaded successfully."
    )

    # -----------------------------------------------------
    # 3. Load classification threshold
    # -----------------------------------------------------

    print(
        "\n[3/4] Loading classification threshold..."
    )

    threshold = load_threshold(
        CONFIG_PATH
    )

    print(
        f"Classification threshold: "
        f"{threshold:.2f}"
    )

    # -----------------------------------------------------
    # 4. Create new customers
    # -----------------------------------------------------

    print(
        "\n[4/4] Predicting customer churn..."
    )

    customers = pd.DataFrame([
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 171.00
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.20,
            "TotalCharges": 3129.60
        },
        {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 6,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.75,
            "TotalCharges": 574.50
        }
    ])

    # -----------------------------------------------------
    # Make predictions
    # -----------------------------------------------------

    for index, customer in customers.iterrows():

        customer_data = pd.DataFrame(
            [customer]
        )

        probability, prediction = (
            predict_customer(
                model,
                preprocessor,
                customer_data,
                threshold
            )
        )

        result = (
            "CHURN"
            if prediction == 1
            else "NO CHURN"
        )

        print("\n" + "-" * 60)

        print(
            f"Customer {index + 1}"
        )

        print(
            f"Churn probability: "
            f"{probability:.2%}"
        )

        print(
            f"Prediction: {result}"
        )

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()