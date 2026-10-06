import json
import numpy as np

from data import (
    load_dataset,
    clean_dataset,
    split_dataset
)

from preprocessing import (
    create_preprocessor,
    preprocess_data,
    save_preprocessor
)

from model import (
    create_model,
    create_callbacks
)

from evaluation import (
    calculate_metrics,
    calculate_roc_auc,
    find_best_f1_threshold
)

from visualization import (
    save_model_architecture,
    plot_training_loss,
    plot_training_accuracy,
    plot_training_auc,
    plot_threshold_analysis,
    plot_confusion_matrix,
    plot_roc_curve
)

from config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    CONFIG_PATH,
    ARCHITECTURE_IMAGE_PATH,
    RESULTS_DIR,
    EPOCHS,
    BATCH_SIZE
)


def main():

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1/14] Loading dataset...")

    dataset = load_dataset()

    print(
        f"Dataset shape: {dataset.shape}"
    )

    # ---------------------------------------------------------
    # 2. Clean dataset
    # ---------------------------------------------------------

    print("\n[2/14] Cleaning dataset...")

    dataset = clean_dataset(dataset)

    print(
        f"Clean dataset shape: {dataset.shape}"
    )

    # ---------------------------------------------------------
    # 3. Split dataset
    # ---------------------------------------------------------

    print("\n[3/14] Splitting dataset...")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = split_dataset(dataset)

    print(
        f"Training samples:   {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Test samples:       {len(X_test)}"
    )

    # ---------------------------------------------------------
    # 4. Create preprocessor
    # ---------------------------------------------------------

    print("\n[4/14] Creating preprocessor...")

    preprocessor = create_preprocessor()

    # ---------------------------------------------------------
    # 5. Preprocess data
    # ---------------------------------------------------------

    print("\n[5/14] Preprocessing data...")

    (
        X_train,
        X_validation,
        X_test
    ) = preprocess_data(
        preprocessor,
        X_train,
        X_validation,
        X_test
    )

    print(
        f"Input features after preprocessing: "
        f"{X_train.shape[1]}"
    )

    # ---------------------------------------------------------
    # Save preprocessing pipeline
    # ---------------------------------------------------------

    print("\nSaving preprocessing pipeline...")

    save_preprocessor(
        preprocessor
    )

    print(
        f"Preprocessor saved to:\n"
        f"{PREPROCESSOR_PATH}"
    )

    # ---------------------------------------------------------
    # 6. Create neural network
    # ---------------------------------------------------------

    print("\n[6/14] Creating neural network...")

    model = create_model(
        X_train.shape[1]
    )

    # ---------------------------------------------------------
    # 7. Display model architecture
    # ---------------------------------------------------------

    print("\n[7/14] Model architecture...")

    model.summary()

    # ---------------------------------------------------------
    # 8. Save model architecture image
    # ---------------------------------------------------------

    print("\n[8/14] Saving model architecture...")

    save_model_architecture(
        model,
        ARCHITECTURE_IMAGE_PATH
    )

    print(
        f"Architecture saved to:\n"
        f"{ARCHITECTURE_IMAGE_PATH}"
    )

    # ---------------------------------------------------------
    # 9. Train model
    # ---------------------------------------------------------

    print("\n[9/14] Training model...")

    callbacks = create_callbacks()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------------------------------------------------
    # 10. Save training plots
    # ---------------------------------------------------------

    print("\n[10/14] Saving training plots...")

    loss_plot_path = (
        RESULTS_DIR
        / "customer_churn_loss.png"
    )

    accuracy_plot_path = (
        RESULTS_DIR
        / "customer_churn_accuracy.png"
    )

    auc_plot_path = (
        RESULTS_DIR
        / "customer_churn_auc.png"
    )

    plot_training_loss(
        history,
        loss_plot_path
    )

    plot_training_accuracy(
        history,
        accuracy_plot_path
    )

    plot_training_auc(
        history,
        auc_plot_path
    )

    print("Training plots saved.")

    # ---------------------------------------------------------
    # 11. Generate validation predictions
    # ---------------------------------------------------------

    print(
        "\n[11/14] Generating validation predictions..."
    )

    y_validation_probability = (
        model.predict(
            X_validation,
            verbose=0
        )
        .ravel()
    )

    # ---------------------------------------------------------
    # 12. Threshold analysis
    # ---------------------------------------------------------

    print(
        "\n[12/14] Performing threshold analysis..."
    )

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    (
        threshold_results,
        best_f1_threshold
    ) = find_best_f1_threshold(
        y_validation,
        y_validation_probability,
        thresholds
    )

    print("\nBest F1 threshold:")

    print(
        f"Threshold: {best_f1_threshold:.2f}"
    )

    threshold_plot_path = (
        RESULTS_DIR
        / "customer_churn_threshold_analysis.png"
    )

    plot_threshold_analysis(
        threshold_results,
        threshold_plot_path
    )

    # The threshold selected using the validation set
    # will be used for the final test evaluation.
    selected_threshold = best_f1_threshold

    print(
        "\nSelected classification threshold:"
    )

    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    # ---------------------------------------------------------
    # Save deployment configuration
    # ---------------------------------------------------------

    deployment_config = {
        "threshold": float(selected_threshold)
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            deployment_config,
            file,
            indent=4
        )

    print(
        f"\nDeployment configuration saved to:\n"
        f"{CONFIG_PATH}"
    )

    # ---------------------------------------------------------
    # 13. Evaluate final model on test set
    # ---------------------------------------------------------

    print(
        "\n[13/14] Evaluating final model on test set..."
    )

    # The test set is used only after:
    #
    # 1. The model has been trained.
    # 2. The preprocessing pipeline has been fitted.
    # 3. The classification threshold has been selected.
    #
    # This prevents information from the test set
    # from influencing the model or threshold selection.

    y_test_probability = (
        model.predict(
            X_test,
            verbose=0
        )
        .ravel()
    )

    test_metrics = calculate_metrics(
        y_test,
        y_test_probability,
        selected_threshold
    )

    test_auc = calculate_roc_auc(
        y_test,
        y_test_probability
    )

    # ---------------------------------------------------------
    # Final test results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Threshold:  {selected_threshold:.2f}"
    )

    print(
        f"Accuracy:   {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:  {test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:     {test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:   {test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:    {test_auc:.4f}"
    )

    # ---------------------------------------------------------
    # Save test confusion matrix
    # ---------------------------------------------------------

    print(
        "\nSaving test confusion matrix..."
    )

    y_test_prediction = (
        y_test_probability
        >= selected_threshold
    ).astype(int)

    confusion_matrix_path = (
        RESULTS_DIR
        / "customer_churn_test_confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_test,
        y_test_prediction,
        confusion_matrix_path,
        "Customer Churn - Test Confusion Matrix"
    )

    print(
        f"Confusion matrix saved to:\n"
        f"{confusion_matrix_path}"
    )

    # ---------------------------------------------------------
    # Save ROC curve
    # ---------------------------------------------------------

    print(
        "\nSaving ROC curve..."
    )

    roc_curve_path = (
        RESULTS_DIR
        / "customer_churn_roc_curve.png"
    )

    plot_roc_curve(
        y_test,
        y_test_probability,
        roc_curve_path
    )

    print(
        f"ROC curve saved to:\n"
        f"{roc_curve_path}"
    )

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------

    print(
        "\nSaving trained model..."
    )

    model.save(
        MODEL_PATH
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    # ---------------------------------------------------------
    # 14. Final model summary
    # ---------------------------------------------------------

    print(
        "\n[14/14] Final model summary..."
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL SUMMARY")
    print("=" * 60)

    print(
        f"Training samples:   {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Test samples:       {len(X_test)}"
    )

    print(
        f"Input features:     {X_train.shape[1]}"
    )

    print(
        "Architecture:       64 -> 32 -> 16 -> 1"
    )

    print(
        f"Batch size:         {BATCH_SIZE}"
    )

    print(
        f"Epochs completed:   "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Best F1 threshold:  "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Test Accuracy:      "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision:     "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall:        "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1:            "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC:       "
        f"{test_auc:.4f}"
    )

    print("\nSaved artifacts:")

    print(
        f"Model:        {MODEL_PATH}"
    )

    print(
        f"Preprocessor: {PREPROCESSOR_PATH}"
    )

    print(
        f"Config:       {CONFIG_PATH}"
    )

    print("\n" + "=" * 60)
    print("EXERCISE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
