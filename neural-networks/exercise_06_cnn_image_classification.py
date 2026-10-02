# ============================================================
# Exercise 06 - Convolutional Neural Network for Image Classification
# ============================================================
#
# Objective:
# Build a Convolutional Neural Network (CNN) from scratch to
# classify images into two categories: cats and dogs.
#
# In this exercise, you will learn how neural networks process
# images and how convolutional layers can automatically detect
# visual patterns such as edges, shapes, and more complex features.
#
# Dataset:
# Use the Microsoft Cats vs Dogs dataset.
#
# Tasks:
#
# 1. Download and load the image dataset.
#
# 2. Resize all images to the same dimensions.
#
# 3. Normalize the pixel values so that they are in the range [0, 1].
#
# 4. Split the dataset into training and validation sets.
#
# 5. Build a Convolutional Neural Network using:
#      - Conv2D
#      - ReLU activation
#      - MaxPooling2D
#      - Flatten
#      - Dense
#      - Dropout
#      - Sigmoid output
#
# 6. Compile the model using:
#      - Adam optimizer
#      - Binary cross-entropy loss
#      - Accuracy as a metric
#
# 7. Train the CNN and monitor:
#      - Training loss
#      - Validation loss
#      - Training accuracy
#      - Validation accuracy
#
# 8. Evaluate the model on unseen test images.
#
# 9. Generate a confusion matrix and calculate:
#      - Accuracy
#      - Precision
#      - Recall
#      - F1-score
#
# 10. Use the trained model to predict whether new images
#     contain a cat or a dog.
#
# 11. Display some test images together with:
#      - The actual class
#      - The predicted class
#      - The prediction probability
#
# CNN architecture:
#
#     Input Image
#          |
#          v
#     Conv2D + ReLU
#          |
#          v
#     MaxPooling2D
#          |
#          v
#     Conv2D + ReLU
#          |
#          v
#     MaxPooling2D
#          |
#          v
#        Flatten
#          |
#          v
#     Dense + ReLU
#          |
#          v
#       Dropout
#          |
#          v
#     Dense + Sigmoid
#          |
#          v
#      Cat / Dog
#
# Important:
# Do NOT use data augmentation or transfer learning in this exercise.
# Those topics will be introduced in later exercises.
#
# The main goal is to understand how a CNN works and why
# convolutional layers are particularly useful for image data.
# ============================================================


from pathlib import Path

import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

SEED = 42


# ============================================================
# 2. Download the Dataset
# ============================================================

dataset_path = kagglehub.dataset_download(
    "shaunthesheep/microsoft-catsvsdogs-dataset"
)

print("Dataset downloaded to:")
print(dataset_path)


# ============================================================
# 3. Locate the Cat and Dog Directories
# ============================================================

dataset_root = Path(dataset_path)

cat_directories = list(dataset_root.rglob("Cat"))
dog_directories = list(dataset_root.rglob("Dog"))

if not cat_directories:
    raise FileNotFoundError(
        f"Could not find the 'Cat' directory inside: {dataset_root}"
    )

if not dog_directories:
    raise FileNotFoundError(
        f"Could not find the 'Dog' directory inside: {dataset_root}"
    )

cat_directory = cat_directories[0]
dog_directory = dog_directories[0]

dataset_root_directory = cat_directory.parent

print("Cat directory:", cat_directory)
print("Dog directory:", dog_directory)
print("Dataset root directory:", dataset_root_directory)



# ============================================================
# 4. Load the Dataset
# ============================================================

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_root_directory,
    labels="inferred",
    label_mode="binary",
    validation_split=VALIDATION_SPLIT,
    subset="both",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_dataset, validation_dataset = dataset


# ============================================================
# 5. Dataset Information
# ============================================================

class_names = train_dataset.class_names

print("Class names:", class_names)


# ============================================================
# 6. Ignore Invalid Images
# ============================================================

train_dataset = train_dataset.apply(
    tf.data.experimental.ignore_errors()
)

validation_dataset = validation_dataset.apply(
    tf.data.experimental.ignore_errors()
)


# ============================================================
# 6. Normalize Pixel Values
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 7. Optimize the Dataset Pipeline
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 8. Inspect the Dataset
# ============================================================

for images, labels in train_dataset.take(1):

    print("Image batch shape:", images.shape)

    print("Label batch shape:", labels.shape)

    print(
        "Minimum pixel value:",
        tf.reduce_min(images).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(images).numpy()
    )

    print(
        "First label:",
        labels[0].numpy()
    )


# ============================================================
# 9. Visualize Sample Images
# ============================================================

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):

    for i in range(9):

        ax = plt.subplot(
            3,
            3,
            i + 1
        )

        plt.imshow(
            images[i].numpy()
        )

        class_index = int(
            labels[i].numpy()[0]
        )

        plt.title(
            class_names[class_index]
        )

        plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 10. Build the CNN Model
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    ),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 11. Display the Model Architecture
# ============================================================

model.summary()


# ============================================================
# 12. Compile the Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 13. Train the CNN
# ============================================================

EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# 14. Plot Training History
# ============================================================

plt.figure(figsize=(12, 5))


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()


plt.tight_layout()

plt.show()

evaluation_results = model.evaluate(
    validation_dataset,
    verbose=1
)

validation_loss = evaluation_results[0]
validation_accuracy = evaluation_results[1]

print()
print("Validation Results")
print("------------------")
print(f"Loss: {validation_loss:.4f}")
print(f"Accuracy: {validation_accuracy:.4f}")


# ============================================================
# 16. Generate Predictions
# ============================================================

y_true = []
y_probabilities = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten()
    )

    y_probabilities.extend(
        predictions.flatten()
    )


# Convert probabilities into binary predictions.
#
# Probability < 0.5 → Cat (0)
# Probability >= 0.5 → Dog (1)

y_true = tf.convert_to_tensor(
    y_true,
    dtype=tf.float32
)

y_probabilities = tf.convert_to_tensor(
    y_probabilities,
    dtype=tf.float32
)

y_pred = tf.cast(
    y_probabilities >= 0.5,
    tf.float32
)


# ============================================================
# 17. Calculate Confusion Matrix
# ============================================================

confusion_matrix = tf.math.confusion_matrix(
    tf.cast(y_true, tf.int32),
    tf.cast(y_pred, tf.int32),
    num_classes=2
)

print()
print("Confusion Matrix")
print("----------------")
print(confusion_matrix.numpy())


# ============================================================
# 18. Extract Confusion Matrix Values
# ============================================================

true_cat = confusion_matrix[0, 0].numpy()
cat_predicted_as_dog = confusion_matrix[0, 1].numpy()

dog_predicted_as_cat = confusion_matrix[1, 0].numpy()
true_dog = confusion_matrix[1, 1].numpy()

print()
print("Confusion Matrix Details")
print("------------------------")
print(f"Correct Cat predictions: {true_cat}")
print(f"Cat predicted as Dog: {cat_predicted_as_dog}")
print(f"Dog predicted as Cat: {dog_predicted_as_cat}")
print(f"Correct Dog predictions: {true_dog}")


# ============================================================
# 19. Calculate Classification Metrics
# ============================================================

true_positive = true_dog
true_negative = true_cat
false_positive = dog_predicted_as_cat
false_negative = cat_predicted_as_dog

total_predictions = (
    true_positive
    + true_negative
    + false_positive
    + false_negative
)

accuracy = (
    (true_positive + true_negative)
    / total_predictions
)

precision = (
    true_positive
    / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


print()
print("Classification Metrics")
print("----------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}")


# ============================================================
# 20. Display the Confusion Matrix
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    confusion_matrix.numpy()
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for row in range(2):

    for column in range(2):

        plt.text(
            column,
            row,
            confusion_matrix[row, column].numpy(),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()


# ============================================================
# 21. Display Sample Predictions
# ============================================================

for images, labels in validation_dataset.take(1):

    probabilities = model.predict(
        images,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    plt.figure(figsize=(12, 12))

    for i in range(9):

        ax = plt.subplot(
            3,
            3,
            i + 1
        )

        plt.imshow(
            images[i].numpy()
        )

        actual_class = class_names[
            int(labels[i].numpy()[0])
        ]

        predicted_class = class_names[
            predictions[i]
        ]

        probability = probabilities[i]

        plt.title(
            f"Actual: {actual_class}\n"
            f"Predicted: {predicted_class}\n"
            f"Dog probability: {probability:.2%}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()