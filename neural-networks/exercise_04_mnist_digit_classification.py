"""
Exercise 04 - Neural Network: MNIST Digit Classification

Objective:
Build a neural network capable of recognizing handwritten digits
from 0 to 9 using the MNIST dataset.

In this exercise, we will learn how to:

* Load and inspect an image dataset.
* Understand the structure of image data.
* Normalize pixel values.
* Transform 28x28 images into a suitable format for a neural network.
* Use a Flatten layer to convert image data into a vector.
* Build a multi-class classification neural network.
* Use ReLU activation functions in hidden layers.
* Use Softmax in the output layer.
* Use Dropout to reduce overfitting.
* Use Early Stopping to stop training when validation performance stops improving.
* Train and evaluate the model.
* Generate predictions for handwritten digits.
* Analyze model performance using a confusion matrix.
* Analyze the most frequent classification errors.
* Visualize incorrect predictions.

Dataset:
MNIST (Modified National Institute of Standards and Technology)

Each sample is a grayscale image of a handwritten digit.

Image size:
28 x 28 pixels

Number of classes:
10 (digits 0 to 9)

The neural network will receive an image as input and predict
the probability of each possible digit.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ============================================================
# 1. Load the MNIST dataset
# ============================================================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# Print the shape of each dataset.
#
# X_train and X_test contain the images.
# y_train and y_test contain the corresponding digit labels.
#
# Expected shapes:
# X_train -> (60000, 28, 28)
# y_train -> (60000,)
# X_test  -> (10000, 28, 28)
# y_test  -> (10000,)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 2. Visualize a sample image
# ============================================================

# Display the first training image.
#
# MNIST images are grayscale, so we use the "gray" color map.

plt.imshow(X_train[0], cmap="gray")

plt.title(f"Label: {y_train[0]}")

plt.axis("off")

plt.show()


# ============================================================
# 3. Normalize pixel values
# ============================================================

# Original MNIST pixel values range from 0 to 255.
#
# Neural networks generally work better when input values
# are kept within a smaller numerical range.
#
# Dividing by 255 transforms the values from:
#
#     0   -> 0.0
#     255 -> 1.0
#
# The resulting range is approximately [0, 1].

X_train = X_train / 255.0
X_test = X_test / 255.0


# ============================================================
# 4. Build the neural network
# ============================================================

# The original image has a shape of:
#
#     28 x 28
#
# Flatten converts it into:
#
#     28 * 28 = 784
#
# input values.
#
# The Dense layers then process these 784 values.

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # First hidden layer.
    # ReLU introduces non-linearity into the network.
    tf.keras.layers.Dense(256, activation="relu"),

    # During training, randomly disables 30% of the neurons.
    #
    # This helps prevent the network from becoming too dependent
    # on specific neurons and can reduce overfitting.
    tf.keras.layers.Dropout(0.30),

    # Second hidden layer.
    tf.keras.layers.Dense(128, activation="relu"),

    # Another Dropout layer for regularization.
    tf.keras.layers.Dropout(0.30),

    # Third hidden layer.
    tf.keras.layers.Dense(64, activation="relu"),

    # Slightly lower dropout rate for the final hidden layer.
    tf.keras.layers.Dropout(0.20),

    # Output layer.
    #
    # MNIST contains 10 classes:
    #
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    #
    # Softmax converts the 10 outputs into probabilities.
    tf.keras.layers.Dense(10, activation="softmax")
])


# ============================================================
# 5. Compile the model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Display the model architecture and number of parameters.

model.summary()


# ============================================================
# 6. Configure Early Stopping
# ============================================================

# Early Stopping monitors the validation loss during training.
#
# If validation loss does not improve for several consecutive
# epochs, training is stopped automatically.
#
# patience=3 means that training can continue for up to
# 3 epochs without improvement before stopping.
#
# restore_best_weights=True restores the weights from the
# epoch that achieved the best validation loss.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 7. Train the model
# ============================================================

# We allow up to 30 epochs.
#
# Early Stopping may stop training earlier if validation
# performance stops improving.
#
# validation_split=0.1 reserves 10% of the training data
# for validation.

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)


# ============================================================
# 8. Analyze the training history
# ============================================================

# Find the epoch with the lowest validation loss.
#
# np.argmin() returns the index of the smallest value.
#
# We add 1 because Python indexes start from 0,
# while epoch numbering starts from 1.

best_epoch = np.argmin(history.history["val_loss"]) + 1

best_val_loss = min(history.history["val_loss"])

best_val_accuracy = history.history["val_accuracy"][best_epoch - 1]

print("\nBest Training Epoch")
print("-------------------")
print("Best Epoch:", best_epoch)
print("Best Validation Loss:", best_val_loss)
print("Validation Accuracy at Best Epoch:", best_val_accuracy)


# ============================================================
# 9. Plot training and validation accuracy
# ============================================================

# Training accuracy shows how well the model performs
# on the data used to update its weights.
#
# Validation accuracy shows how well the model performs
# on data that was not used to update the weights.

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()


# ============================================================
# 10. Plot training and validation loss
# ============================================================

# Training loss measures the error on the training data.
#
# Validation loss measures the error on the validation data.
#
# A validation loss that starts increasing while training loss
# continues decreasing can be a sign of overfitting.

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()


# ============================================================
# 11. Evaluate the model on the test set
# ============================================================

# The test set was never used to train the model.
#
# It provides an independent evaluation of the final model.

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print("\nFinal Test Results")
print("------------------")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 12. Generate predictions
# ============================================================

# model.predict() returns the probability distribution
# for each of the 10 possible digits.
#
# Example:
#
# [0.01, 0.00, 0.02, 0.03, 0.90, ...]
#
# The highest probability represents the predicted digit.

predictions = model.predict(X_test)

# np.argmax() returns the index of the highest probability.
#
# This converts the probability vector into the predicted
# digit class.

y_pred = np.argmax(predictions, axis=1)


# ============================================================
# 13. Inspect one prediction
# ============================================================

print("\nPrediction probabilities:")
print(predictions[0])

print("Predicted digit:", predictions[0].argmax())
print("Actual digit:", y_test[0])


# Visualize the prediction.

plt.imshow(X_test[0], cmap="gray")

plt.title(
    f"Predicted: {predictions[0].argmax()} | Actual: {y_test[0]}"
)

plt.axis("off")

plt.show()


# ============================================================
# 14. Generate the confusion matrix
# ============================================================

# The confusion matrix shows how often each actual digit
# was classified as each predicted digit.
#
# Rows    -> actual labels
# Columns -> predicted labels
#
# A perfect model would have values concentrated
# on the main diagonal.

confusion_matrix = tf.math.confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(confusion_matrix)


# Visualize the confusion matrix.

plt.figure(figsize=(10, 8))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("MNIST Confusion Matrix")

plt.show()


# ============================================================
# 15. Find incorrect predictions
# ============================================================

# np.where() returns the indexes where the prediction
# does not match the actual label.

wrong_predictions = np.where(y_pred != y_test)[0]

print(
    "\nNumber of incorrect predictions:",
    len(wrong_predictions)
)


# ============================================================
# 16. Visualize the first incorrect prediction
# ============================================================

index = wrong_predictions[0]

plt.imshow(X_test[index], cmap="gray")

plt.title(
    f"Predicted: {y_pred[index]} | Actual: {y_test[index]}"
)

plt.axis("off")

plt.show()


# ============================================================
# 17. Visualize multiple incorrect predictions
# ============================================================

# Display the first 10 incorrectly classified images.
#
# This helps us visually inspect the types of images
# that are difficult for the model.

num_images = 10

plt.figure(figsize=(12, 6))

for i in range(num_images):

    index = wrong_predictions[i]

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[index], cmap="gray")

    plt.title(
        f"Pred: {y_pred[index]} | Real: {y_test[index]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 18. Analyze the most frequent classification errors
# ============================================================

# The confusion matrix allows us to identify which pairs
# of digits are most frequently confused.
#
# For example:
#
# Actual 3 -> Predicted 5
#
# means that the model incorrectly classified some images
# of the digit 3 as the digit 5.

print("\nClassification Errors")
print("----------------------")

for actual_digit in range(10):

    for predicted_digit in range(10):

        # Ignore the main diagonal because those are
        # correct predictions.

        if actual_digit != predicted_digit:

            count = confusion_matrix[
                actual_digit,
                predicted_digit
            ]

            if count > 0:

                print(
                    f"Actual {actual_digit} -> "
                    f"Predicted {predicted_digit}: {count}"
                )