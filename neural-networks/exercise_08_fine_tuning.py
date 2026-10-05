"""
Exercise 08 - Fine-Tuning a Pretrained CNN

Goal:
Improve an image classification model by fine-tuning a pretrained
MobileNetV2 network.

Tasks:
1. Load the cats and dogs image dataset.
2. Load MobileNetV2 with pretrained ImageNet weights.
3. Compare feature extraction with fine-tuning.
4. Freeze the pretrained layers initially.
5. Unfreeze selected layers and retrain with a lower learning rate.
6. Compare validation loss and accuracy before and after fine-tuning.
7. Analyze overfitting and model generalization.
8. Predict completely new images not used during training.
9. Analyze prediction probabilities.

Expected outcome:
Understand how fine-tuning adapts pretrained neural networks
to a specific image classification task.
"""


from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ======================================================
# Configuration
# ======================================================

IMG_HEIGHT = 160
IMG_WIDTH = 160

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

SEED = 42

EPOCHS = 10

LEARNING_RATE = 0.001

FINE_TUNE_AT = 100

FINE_TUNE_LEARNING_RATE = 0.00001

TEST_IMAGES_DIR = Path("test_images")


# ======================================================
# Create Neural Network
# ======================================================

def add_neural_network(
    trainable: bool,
    learning_rate: float,
    fine_tune_at: int = 100,
    fine_tune_learning_rate: float = 0.00001
) -> dict:

    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        ),
        include_top=False,
        weights="imagenet"
    )

    # Freeze the entire pre-trained network initially
    base_model.trainable = False

    # ==================================================
    # Fine-Tuning
    # ==================================================

    if trainable:

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    # MobileNetV2 preprocessing
    preprocess_input = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )

    # Input layer
    inputs = tf.keras.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )

    # Preprocess input
    x = preprocess_input(inputs)

    # Keep BatchNormalization layers in inference mode
    x = base_model(
        x,
        training=False
    )

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3
    )(x)

    # Binary classification
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    # Create model
    model = tf.keras.Model(
        inputs,
        outputs
    )

    # ==================================================
    # Optimizer
    # ==================================================

    if trainable:

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=fine_tune_learning_rate
        )

    else:

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ==================================================
    # Training
    # ==================================================

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # ==================================================
    # Evaluation
    # ==================================================

    validation_loss, validation_accuracy = model.evaluate(
        val_ds,
        verbose=1
    )

    # Count trainable parameters
    trainable_parameters = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    # Count total parameters
    total_parameters = model.count_params()

    return {
        "base_model": base_model,
        "model": model,
        "history": history,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "trainable": trainable,
        "fine_tune_at": fine_tune_at,
        "learning_rate": (
            fine_tune_learning_rate
            if trainable
            else learning_rate
        ),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters
    }


# ======================================================
# Visualize Results
# ======================================================

def visualize_results(
    title: str,
    accuracy: list[float],
    val_accuracy: list[float],
    loss: list[float],
    val_loss: list[float],
    model: tf.keras.Model,
    dataset: tf.data.Dataset
) -> None:

    # ==================================================
    # Predictions
    # ==================================================

    predictions = model.predict(
        dataset,
        verbose=0
    )

    predicted_classes = (
        predictions >= 0.5
    ).astype(int).flatten()

    # ==================================================
    # True Labels
    # ==================================================

    true_classes = np.concatenate([
        labels.numpy().flatten()
        for _, labels in dataset
    ]).astype(int)

    # ==================================================
    # Confusion Matrix
    # ==================================================

    matrix = confusion_matrix(
        true_classes,
        predicted_classes
    )

    # ==================================================
    # Epochs
    # ==================================================

    epochs = range(
        1,
        len(accuracy) + 1
    )

    # ==================================================
    # Create Figure
    # ==================================================

    plt.figure(
        figsize=(18, 6)
    )

    # ==================================================
    # Accuracy
    # ==================================================

    plt.subplot(
        1,
        3,
        1
    )

    plt.plot(
        epochs,
        accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.title(
        f"{title} - Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    # Show complete accuracy range
    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(
        True
    )

    # ==================================================
    # Loss
    # ==================================================

    plt.subplot(
        1,
        3,
        2
    )

    plt.plot(
        epochs,
        loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.title(
        f"{title} - Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    # ==================================================
    # Confusion Matrix
    # ==================================================

    plt.subplot(
        1,
        3,
        3
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Cat",
            "Dog"
        ]
    )

    display.plot(
        ax=plt.gca(),
        colorbar=False
    )

    plt.title(
        f"{title} - Confusion Matrix"
    )

    # ==================================================
    # Show Figure
    # ==================================================

    plt.tight_layout()

    plt.show()


# ======================================================
# Predict New Image
# ======================================================

def predict_image(
    model: tf.keras.Model,
    image_path: Path
) -> None:

    # Check if image exists
    if not image_path.exists():

        print()
        print(
            f"Image not found: {image_path}"
        )

        return

    # ==================================================
    # Load Image
    # ==================================================

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(
            IMG_HEIGHT,
            IMG_WIDTH
        )
    )

    # ==================================================
    # Convert Image to Array
    # ==================================================

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # Add batch dimension
    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    # ==================================================
    # Prediction
    # ==================================================

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # ==================================================
    # Calculate Probabilities
    # ==================================================

    dog_probability = float(
        prediction
    )

    cat_probability = 1.0 - dog_probability

    # ==================================================
    # Determine Class
    # ==================================================

    if dog_probability >= 0.5:

        predicted_class = "Dog"

        predicted_probability = (
            dog_probability
        )

    else:

        predicted_class = "Cat"

        predicted_probability = (
            cat_probability
        )

    # ==================================================
    # Print Results
    # ==================================================

    print()
    print(
        "========================"
    )

    print(
        "Image Prediction"
    )

    print(
        "========================"
    )

    print(
        f"Image: {image_path}"
    )

    print(
        f"Predicted Class: "
        f"{predicted_class}"
    )

    print(
        f"Cat Probability: "
        f"{cat_probability:.4f}"
    )

    print(
        f"Dog Probability: "
        f"{dog_probability:.4f}"
    )

    print(
        f"Prediction Confidence: "
        f"{predicted_probability:.4f}"
    )

    # ==================================================
    # Visualize Image
    # ==================================================

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image
    )

    plt.title(
        f"{predicted_class}\n"
        f"Cat: {cat_probability:.2%} | "
        f"Dog: {dog_probability:.2%}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()


# ======================================================
# Analyze New Image
# ======================================================

def analyze_new_images(
    model: tf.keras.Model,
    images_directory: Path
) -> None:

    if not images_directory.exists():

        print()
        print(
            f"Directory not found: "
            f"{images_directory}"
        )

        print()
        print(
            "Create the directory and add "
            "new images before running predictions."
        )

        return

    # Supported image extensions
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    # Find images
    image_paths = [
        path
        for path in images_directory.iterdir()
        if path.suffix.lower()
        in image_extensions
    ]

    if not image_paths:

        print()
        print(
            f"No images found in "
            f"{images_directory}"
        )

        return

    print()
    print(
        "========================"
    )

    print(
        "Unseen Images"
    )

    print(
        "========================"
    )

    print(
        f"Found {len(image_paths)} image(s)."
    )

    # Predict every image
    for image_path in image_paths:

        predict_image(
            model=model,
            image_path=image_path
        )


# ======================================================
# Download Dataset
# ======================================================

dataset_path = kagglehub.dataset_download(
    "shaunthesheep/microsoft-catsvsdogs-dataset"
)

dataset_path = Path(
    dataset_path
)


# ======================================================
# Find Cat and Dog Directories
# ======================================================

cat_dir = list(
    dataset_path.rglob("Cat")
)

dog_dir = list(
    dataset_path.rglob("Dog")
)

if not cat_dir or not dog_dir:

    raise FileNotFoundError(
        "Could not find Cat and Dog directories."
    )


main_dir = cat_dir[0].parent


print()
print(
    f"Main dataset directory: {main_dir}"
)

print(
    f"Downloaded dataset directory: {dataset_path}"
)


# ======================================================
# Create Training and Validation Datasets
# ======================================================

train_ds, val_ds = (
    tf.keras.utils.image_dataset_from_directory(
        main_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="both",
        seed=SEED,
        image_size=(
            IMG_HEIGHT,
            IMG_WIDTH
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
)


# ======================================================
# Ignore Corrupted Images
# ======================================================

train_ds = train_ds.apply(
    tf.data.experimental.ignore_errors()
)

val_ds = val_ds.apply(
    tf.data.experimental.ignore_errors()
)


# ======================================================
# Transfer Learning
# ======================================================

print()
print(
    "========================"
)

print(
    "Starting Transfer Learning"
)

print(
    "========================"
)


transfer_learning = add_neural_network(
    trainable=False,
    learning_rate=LEARNING_RATE
)


# ======================================================
# Transfer Learning Results
# ======================================================

print()
print(
    "========================"
)

print(
    "Transfer Learning Results"
)

print(
    "========================"
)

print(
    f"Validation Loss: "
    f"{transfer_learning['validation_loss']:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{transfer_learning['validation_accuracy']:.4f}"
)

print(
    f"Trainable Parameters: "
    f"{transfer_learning['trainable_parameters']:,}"
)

print(
    f"Total Parameters: "
    f"{transfer_learning['total_parameters']:,}"
)


# ======================================================
# Visualize Transfer Learning
# ======================================================

transfer_history = (
    transfer_learning["history"].history
)

visualize_results(
    title="Transfer Learning",
    accuracy=transfer_history["accuracy"],
    val_accuracy=transfer_history["val_accuracy"],
    loss=transfer_history["loss"],
    val_loss=transfer_history["val_loss"],
    model=transfer_learning["model"],
    dataset=val_ds
)


# ======================================================
# Fine-Tuning
# ======================================================

print()
print(
    "========================"
)

print(
    "Starting Fine-Tuning"
)

print(
    "========================"
)


fine_tuning = add_neural_network(
    trainable=True,
    learning_rate=LEARNING_RATE,
    fine_tune_at=FINE_TUNE_AT,
    fine_tune_learning_rate=FINE_TUNE_LEARNING_RATE
)


# ======================================================
# Fine-Tuning Results
# ======================================================

print()
print(
    "========================"
)

print(
    "Fine-Tuning Results"
)

print(
    "========================"
)

print(
    f"Fine-Tuning starts at layer: "
    f"{fine_tuning['fine_tune_at']}"
)

print(
    f"Learning Rate: "
    f"{fine_tuning['learning_rate']}"
)

print(
    f"Validation Loss: "
    f"{fine_tuning['validation_loss']:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{fine_tuning['validation_accuracy']:.4f}"
)

print(
    f"Trainable Parameters: "
    f"{fine_tuning['trainable_parameters']:,}"
)

print(
    f"Total Parameters: "
    f"{fine_tuning['total_parameters']:,}"
)


# ======================================================
# Visualize Fine-Tuning
# ======================================================

fine_tuning_history = (
    fine_tuning["history"].history
)

visualize_results(
    title="Fine-Tuning",
    accuracy=fine_tuning_history["accuracy"],
    val_accuracy=fine_tuning_history["val_accuracy"],
    loss=fine_tuning_history["loss"],
    val_loss=fine_tuning_history["val_loss"],
    model=fine_tuning["model"],
    dataset=val_ds
)


# ======================================================
# Model Comparison
# ======================================================

transfer_accuracy = (
    transfer_learning["validation_accuracy"]
)

fine_tuning_accuracy = (
    fine_tuning["validation_accuracy"]
)

transfer_loss = (
    transfer_learning["validation_loss"]
)

fine_tuning_loss = (
    fine_tuning["validation_loss"]
)


accuracy_difference = (
    fine_tuning_accuracy
    - transfer_accuracy
)

loss_difference = (
    fine_tuning_loss
    - transfer_loss
)


print()
print(
    "========================"
)

print(
    "Model Comparison"
)

print(
    "========================"
)

print(
    f"Transfer Learning Accuracy: "
    f"{transfer_accuracy:.4f}"
)

print(
    f"Fine-Tuning Accuracy: "
    f"{fine_tuning_accuracy:.4f}"
)

print()

print(
    f"Transfer Learning Loss: "
    f"{transfer_loss:.4f}"
)

print(
    f"Fine-Tuning Loss: "
    f"{fine_tuning_loss:.4f}"
)

print()

print(
    f"Accuracy Difference: "
    f"{accuracy_difference:+.4f}"
)

print(
    f"Loss Difference: "
    f"{loss_difference:+.4f}"
)


# ======================================================
# Analyze Results
# ======================================================

print()
print(
    "========================"
)

print(
    "Fine-Tuning Analysis"
)

print(
    "========================"
)


if accuracy_difference > 0:

    print(
        "Fine-tuning improved "
        "validation accuracy."
    )

elif accuracy_difference < 0:

    print(
        "Fine-tuning reduced "
        "validation accuracy."
    )

else:

    print(
        "Fine-tuning produced "
        "the same validation accuracy."
    )


if loss_difference < 0:

    print(
        "Fine-tuning reduced "
        "validation loss."
    )

elif loss_difference > 0:

    print(
        "Fine-tuning increased "
        "validation loss."
    )

else:

    print(
        "Fine-tuning produced "
        "the same validation loss."
    )


# ======================================================
# Overfitting Analysis
# ======================================================

final_training_accuracy = (
    fine_tuning_history["accuracy"][-1]
)

final_validation_accuracy = (
    fine_tuning_history["val_accuracy"][-1]
)

accuracy_gap = (
    final_training_accuracy
    - final_validation_accuracy
)


print()
print(
    "========================"
)

print(
    "Overfitting Analysis"
)

print(
    "========================"
)

print(
    f"Final Training Accuracy: "
    f"{final_training_accuracy:.4f}"
)

print(
    f"Final Validation Accuracy: "
    f"{final_validation_accuracy:.4f}"
)

print(
    f"Training/Validation Gap: "
    f"{accuracy_gap:.4f}"
)


if accuracy_gap > 0.05:

    print(
        "Possible overfitting detected."
    )

else:

    print(
        "No significant overfitting detected."
    )


# ======================================================
# Predictions on Completely New Images
# ======================================================

print()
print(
    "========================"
)

print(
    "Predictions on Unseen Images"
)

print(
    "========================"
)

analyze_new_images(
    model=fine_tuning["model"],
    images_directory=TEST_IMAGES_DIR
)
