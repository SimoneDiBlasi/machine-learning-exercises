"""
Exercise 05 - Neural Network: Overfitting and Regularization

Objective:
Understand how overfitting affects neural networks and learn
how regularization techniques can improve model generalization.

In this exercise, we will learn how to:

* Understand what overfitting is.
* Identify overfitting using training and validation metrics.
* Build a neural network that intentionally overfits the training data.
* Compare training performance with validation performance.
* Visualize training and validation loss.
* Visualize training and validation accuracy.
* Understand the effect of model complexity.
* Use Dropout to reduce overfitting.
* Use Early Stopping to prevent unnecessary training.
* Compare models with and without regularization.
* Evaluate the final model on unseen test data.

Dataset:
We will use the Fashion MNIST dataset.

Fashion MNIST contains grayscale images representing
different categories of clothing and accessories.

Image size:
28 x 28 pixels

Number of classes:
10

Classes:
0 - T-shirt / Top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle Boot

Main goal:
Instead of focusing only on achieving high training accuracy,
we will learn how to build a model that generalizes well
to data it has never seen before.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Print the shape of each dataset.

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# ============================================================
# 2. Visualize sample images
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[i], cmap="gray")

    plt.title(class_names[y_train[i]])

    plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 3. Analyze the class distribution
# ============================================================

# Count how many samples belong to each class.

class_counts = np.bincount(y_train)

for class_index, count in enumerate(class_counts):
    print(
        f"{class_index} - {class_names[class_index]}: {count}"
    )

plt.figure(figsize=(10, 5))

plt.bar(
    class_names,
    class_counts
)

plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Fashion MNIST Class Distribution")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 4. Normalize pixel values
# ============================================================

# Convert pixel values from the range [0, 255]
# to the range [0, 1].

X_train = X_train / 255.0
X_test = X_test / 255.0


# ============================================================
# 5. Build a high-capacity neural network
# ============================================================

# This model is intentionally larger than necessary.
#
# The goal is to observe what happens when a neural network
# has enough capacity to memorize the training data.
#
# We will later compare this model with regularized versions.

model = tf.keras.Sequential([

    # Convert each 28x28 image into a vector of 784 values.
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Large hidden layers give the network a high capacity.
    tf.keras.layers.Dense(512, activation="relu"),

    tf.keras.layers.Dense(256, activation="relu"),

    tf.keras.layers.Dense(128, activation="relu"),

    # Output layer:
    # 10 neurons because Fashion MNIST has 10 classes.
    #
    # Softmax converts the outputs into class probabilities.
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. Train the model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)

# ============================================================
# 8. Visualize training and validation performance
# ============================================================

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
# 9. Evaluate the model on unseen test data
# ============================================================

# The test set contains images that the model has never seen
# during training or validation.

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print("\nTest Results")
print("------------")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ============================================================
# 10. Compare training, validation and test performance
# ============================================================

final_training_accuracy = history.history["accuracy"][-1]

final_validation_accuracy = history.history["val_accuracy"][-1]

print("\nPerformance Comparison")
print("----------------------")
print("Final Training Accuracy:", final_training_accuracy)
print("Final Validation Accuracy:", final_validation_accuracy)
print("Test Accuracy:", test_accuracy)

# ============================================================
# 11. Build a regularized neural network
# ============================================================

# This model uses the same architecture as the previous model,
# but adds Dropout layers to reduce overfitting.

regularized_model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(10, activation="softmax")
])

# ============================================================
# 12. Compile the regularized model
# ============================================================

regularized_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

regularized_model.summary()

# ============================================================
# 13. Train the regularized model
# ============================================================

regularized_history = regularized_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)

# ============================================================
# 14. Visualize regularized model performance
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    regularized_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    regularized_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Regularized Model - Training vs Validation Accuracy")

plt.legend()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    regularized_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    regularized_history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Regularized Model - Training vs Validation Loss")

plt.legend()

plt.show()


# ============================================================
# 15. Configure Early Stopping
# ============================================================

# Early Stopping monitors the validation loss.
#
# If the validation loss does not improve for 3 consecutive
# epochs, training will stop automatically.
#
# restore_best_weights=True restores the model weights from
# the epoch with the best validation loss.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ============================================================
# 16. Build the final regularized model
# ============================================================

final_model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(10, activation="softmax")
])


final_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 17. Train the final model
# ============================================================

final_history = final_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# ============================================================
# 18. Analyze Early Stopping
# ============================================================

epochs_trained = len(final_history.history["loss"])

best_epoch = np.argmin(
    final_history.history["val_loss"]
) + 1

best_val_loss = min(
    final_history.history["val_loss"]
)

print("\nEarly Stopping Results")
print("----------------------")
print("Epochs actually trained:", epochs_trained)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

# ============================================================
# 19. Evaluate the final model
# ============================================================

final_test_loss, final_test_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print("\nFinal Model Results")
print("-------------------")
print("Test Loss:", final_test_loss)
print("Test Accuracy:", final_test_accuracy)

# ============================================================
# 20. Compare all three models
# ============================================================

model_a_test_loss, model_a_test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

model_b_test_loss, model_b_test_accuracy = regularized_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

model_c_test_loss, model_c_test_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Comparison")
print("----------------")

print(
    f"Model A - No Regularization: "
    f"{model_a_test_accuracy:.4f}"
)

print(
    f"Model B - Dropout: "
    f"{model_b_test_accuracy:.4f}"
)

print(
    f"Model C - Dropout + Early Stopping: "
    f"{model_c_test_accuracy:.4f}"
)

print("\nFinal Training Accuracy")
print("-----------------------")

print(
    "Model A:",
    history.history["accuracy"][-1]
)

print(
    "Model B:",
    regularized_history.history["accuracy"][-1]
)

print(
    "Model C:",
    final_history.history["accuracy"][-1]
)
