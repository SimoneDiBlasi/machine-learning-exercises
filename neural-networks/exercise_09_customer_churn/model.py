import tensorflow as tf

from config import (
    LEARNING_RATE,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
)


def create_model(input_features):
    """
    Create and compile the customer churn neural network.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_features,)
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            DROPOUT_RATE_1
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            DROPOUT_RATE_2
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


def create_callbacks():
    """
    Create callbacks used during model training.
    """

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    return [
        reduce_lr,
        early_stopping
    ]
