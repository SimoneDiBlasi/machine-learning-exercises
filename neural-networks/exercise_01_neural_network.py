"""
Exercise 01 - Neural Network: Customer Purchase Prediction

Objective:
    Build your first Neural Network using TensorFlow/Keras to predict
    whether a customer will purchase a product.

Dataset:
    Each customer is described by the following features:

    - Age
    - Annual Income
    - Website Time
    - Previous Purchases
    - Pages Visited

Target:
    - 0 = Customer does not purchase
    - 1 = Customer purchases

Tasks:
    1. Create a dataset containing customer information and purchase decisions.
    2. Separate the dataset into input features (X) and target variable (y).
    3. Split the dataset into training set (80%) and test set (20%).
    4. Standardize the input features using StandardScaler.
    5. Build a Neural Network using TensorFlow/Keras.
    6. Compile the model using Adam, Binary Crossentropy and Accuracy.
    7. Train the Neural Network for 100 epochs.
    8. Use 20% of the training data as validation data.
    9. Plot training and validation accuracy.
    10. Plot training and validation loss.
    11. Evaluate the trained model using the test set.
    12. Predict whether a new customer will purchase the product.
    13. Display the predicted purchase probability and class.
    14. Create a confusion matrix.
    15. Print a classification report.

Bonus:
    Experiment with different Neural Network architectures:

    Network A:
        5 → 4 → 1

    Network B:
        5 → 16 → 8 → 1

    Network C:
        5 → 64 → 32 → 16 → 1
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. Create the dataset
# ============================================================

X = np.array([
    [22, 22000, 10, 0, 2],
    [25, 28000, 15, 1, 3],
    [28, 32000, 20, 1, 4],
    [31, 35000, 25, 2, 5],
    [35, 40000, 30, 2, 6],
    [38, 45000, 35, 3, 7],
    [42, 50000, 40, 4, 8],
    [45, 55000, 45, 5, 9],
    [50, 60000, 50, 6, 10],
    [55, 65000, 55, 7, 11],
    [60, 70000, 60, 8, 12],
    [65, 75000, 65, 9, 13],
    [27, 30000, 18, 1, 3],
    [33, 38000, 28, 2, 5],
    [40, 48000, 38, 3, 7],
    [48, 58000, 48, 5, 9],
    [53, 62000, 52, 6, 10],
    [58, 68000, 58, 7, 11],
    [24, 25000, 12, 0, 2],
    [37, 43000, 32, 3, 6]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    1
])


print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("First customer:", X[0])
print("First target:", y[0])


# ============================================================
# 2. Split the dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 3. Standardize the input features
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nX_train after scaling:")
print(X_train)

print("\nX_test after scaling:")
print(X_test)


# ============================================================
# 4. Build the Neural Network
# ============================================================

def build_model(hidden_layers):
    """
    Build and compile a Neural Network.

    Parameters:
        hidden_layers: list containing the number of neurons
                       for each hidden layer.

    Example:
        [16, 8, 4]

    Creates:
        Input(5) → Dense(16) → Dense(8) → Dense(4) → Dense(1)
    """

    model = tf.keras.Sequential()

    # Input layer
    model.add(
        tf.keras.layers.Input(shape=(5,))
    )

    # Hidden layers
    for neurons in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                neurons,
                activation="relu"
            )
        )

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    # Configure the training process
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. Choose the Neural Network architecture
# ============================================================

architecture = [16, 8, 4]

model = build_model(architecture)


# Display the model architecture
print("\nModel architecture:")
model.summary()


# ============================================================
# 6. Train the Neural Network
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# 7. Plot Training and Validation Accuracy
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot Training and Validation Loss
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.tight_layout()
plt.show()


# ============================================================
# 9. Evaluate the model using the test set
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 10. Predict a new customer
# ============================================================

new_customer = np.array([
    [36, 42000, 35, 3, 7]
])

# Apply the same scaler used for the training data
new_customer_scaled = scaler.transform(new_customer)

# Predict purchase probability
purchase_probability = model.predict(
    new_customer_scaled,
    verbose=0
)

probability = purchase_probability[0][0]

# Convert probability into a class
predicted_class = 1 if probability >= 0.5 else 0

print("\nNew Customer Prediction")
print("Purchase probability:", probability)
print("Predicted class:", predicted_class)


# ============================================================
# 11. Generate predictions for the test set
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
)

# Convert probabilities into classes using a 0.5 threshold
y_pred = (y_probability >= 0.5).astype(int)


# ============================================================
# 12. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 13. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 14. Experiment with different architectures
# ============================================================

architectures = {
    "Network A": [4],
    "Network B": [16, 8],
    "Network C": [64, 32, 16]
}

print("\nArchitecture experiments:")

for name, architecture in architectures.items():

    print("\n" + "=" * 50)
    print(name)
    print("Architecture:", architecture)
    print("=" * 50)

    experiment_model = build_model(architecture)

    experiment_model.summary()

    experiment_history = experiment_model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_split=0.2,
        verbose=0
    )

    experiment_loss, experiment_accuracy = experiment_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    final_training_accuracy = experiment_history.history["accuracy"][-1]
    final_validation_accuracy = experiment_history.history["val_accuracy"][-1]

    final_training_loss = experiment_history.history["loss"][-1]
    final_validation_loss = experiment_history.history["val_loss"][-1]

    print("\nResults:")
    print("Training Accuracy:", final_training_accuracy)
    print("Validation Accuracy:", final_validation_accuracy)
    print("Test Accuracy:", experiment_accuracy)

    print("Training Loss:", final_training_loss)
    print("Validation Loss:", final_validation_loss)
    print("Test Loss:", experiment_loss)