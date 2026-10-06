import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_probability,
    threshold=0.50
):
    """
    Calculate classification metrics using a probability threshold.
    """

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_prediction
    )

    precision = precision_score(
        y_true,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_prediction,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_prediction
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }


# ============================================================
# ROC-AUC
# ============================================================

def calculate_roc_auc(
    y_true,
    y_probability
):
    """
    Calculate ROC-AUC using prediction probabilities.
    """

    return roc_auc_score(
        y_true,
        y_probability
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def find_best_f1_threshold(
    y_true,
    y_probability,
    thresholds
):
    """
    Evaluate multiple thresholds and return:

    1. A list containing the metrics for every threshold.
    2. The threshold with the highest F1 score.
    """

    threshold_results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_probability,
            threshold
        )

        threshold_results.append(
            metrics
        )

    best_result = max(
        threshold_results,
        key=lambda result: result["f1"]
    )

    best_f1_threshold = (
        best_result["threshold"]
    )

    print(
        "\nThreshold Analysis"
    )

    print(
        "-" * 60
    )

    for result in threshold_results:

        print(
            f"Threshold: {result['threshold']:.2f} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"F1: {result['f1']:.4f}"
        )

    print("\nBest F1 threshold:")

    print(f"Threshold: {best_f1_threshold:.2f}")

    print(f"F1 Score: {best_result['f1']:.4f}")

    return (
        threshold_results,
        best_f1_threshold
    )


# ============================================================
# BUSINESS THRESHOLD
# ============================================================

def find_business_threshold(
    threshold_results,
    minimum_recall=0.60
):
    """
    Select a threshold that satisfies a minimum recall.

    Among thresholds that satisfy the minimum recall,
    the threshold with the highest precision is selected.

    If no threshold satisfies the requirement,
    the threshold with the highest recall is selected.
    """

    valid_results = [
        result
        for result in threshold_results
        if result["recall"] >= minimum_recall
    ]

    if valid_results:

        selected_result = max(
            valid_results,
            key=lambda result: result["precision"]
        )

    else:

        selected_result = max(
            threshold_results,
            key=lambda result: result["recall"]
        )

    selected_threshold = (
        selected_result["threshold"]
    )

    print("\nSelected business threshold:")

    print(f"Threshold: {selected_threshold:.2f}")

    print(f"Precision: {selected_result['precision']:.4f}")

    print(f"Recall: {selected_result['recall']:.4f}")

    print(f"F1 Score: {selected_result['f1']:.4f}")

    return selected_threshold
