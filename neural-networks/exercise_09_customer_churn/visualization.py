import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_curve
)

from tensorflow.keras.utils import plot_model


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def save_model_architecture(
    model,
    output_path
):
    """
    Save a visual representation of the neural network.
    """

    plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        dpi=120
    )


# ============================================================
# TRAINING LOSS
# ============================================================

def plot_training_loss(
    history,
    output_path
):
    """
    Plot and save training and validation loss.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training Loss")

    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=120, bbox_inches="tight")

    plt.close()


# ============================================================
# TRAINING ACCURACY
# ============================================================

def plot_training_accuracy(
    history,
    output_path
):
    """
    Plot and save training and validation accuracy.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# TRAINING AUC
# ============================================================

def plot_training_auc(
    history,
    output_path
):
    """
    Plot and save training and validation AUC.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["auc"], label="Training AUC")
    plt.plot(history.history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training and Validation AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def plot_threshold_analysis(
    threshold_results,
    output_path
):
    """
    Plot accuracy, precision, recall and F1
    for different classification thresholds.
    """

    thresholds = [
        result["threshold"]
        for result in threshold_results
    ]

    accuracy = [
        result["accuracy"]
        for result in threshold_results
    ]

    precision = [
        result["precision"]
        for result in threshold_results
    ]

    recall = [
        result["recall"]
        for result in threshold_results
    ]

    f1 = [
        result["f1"]
        for result in threshold_results
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        thresholds,
        accuracy,
        marker="o",
        label="Accuracy"
    )

    plt.plot(
        thresholds,
        precision,
        marker="o",
        label="Precision"
    )

    plt.plot(
        thresholds,
        recall,
        marker="o",
        label="Recall"
    )

    plt.plot(
        thresholds,
        f1,
        marker="o",
        label="F1 Score"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

# ============================================================
# CONFUSION MATRIX
# ============================================================


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    title="Confusion Matrix"
):
    """
    Plot and save a confusion matrix.

    Parameters
    ----------
    y_true:
        True target values.

    y_pred:
        Predicted target values.

    output_path:
        Path where the image will be saved.

    title:
        Title displayed above the confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    class_names = [
        "No Churn",
        "Churn"
    ]

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = (
        matrix.max() / 2
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                matrix[i, j],
                horizontalalignment="center",
                color="white"
                if matrix[i, j] > threshold
                else "black"
            )

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true,
    y_probability,
    output_path
):
    """
    Plot and save the ROC curve.
    """

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            y_probability
        )
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label="ROC Curve"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Receiver Operating Characteristic (ROC) Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(output_path, dpi=120, bbox_inches="tight")

    plt.close()
