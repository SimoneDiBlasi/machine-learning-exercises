import json
import joblib
import tensorflow as tf


def load_model(model_path):
    """
    Load the trained neural network from disk.
    """

    return tf.keras.models.load_model(
        model_path
    )


def load_preprocessor(preprocessor_path):
    """
    Load the fitted preprocessing pipeline from disk.
    """

    return joblib.load(
        preprocessor_path
    )


def load_threshold(config_path):
    """
    Load the classification threshold from disk.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        configuration = json.load(file)

    return configuration["threshold"]


def predict_customer(
    model,
    preprocessor,
    customer_data,
    threshold
):
    """
    Predict customer churn probability and class.
    """

    # Apply the same preprocessing used during training.
    transformed_data = preprocessor.transform(
        customer_data
    )

    # Generate the churn probability.
    probability = (
        model.predict(
            transformed_data,
            verbose=0
        )
        .ravel()[0]
    )

    # Convert the probability into a binary prediction.
    prediction = int(
        probability >= threshold
    )

    return probability, prediction