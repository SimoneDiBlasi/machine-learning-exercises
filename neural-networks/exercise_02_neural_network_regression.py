"""
Exercise 02 - Neural Network Regression: House Price Prediction

Objective:
Build a neural network using TensorFlow/Keras to predict the price of a house
based on its characteristics.

Dataset:
Each house is described by the following features:

- Surface Area (m²)
- Bedrooms
- Bathrooms
- House Age (years)
- Distance from City Center (km)

Target:
- House Price (€)

Tasks:

1. Create the dataset using NumPy.

2. Split the dataset into training and test sets using train_test_split.

3. Standardize the input features and target using StandardScaler.
   Remember to fit the scalers only on the training data.

4. Build a neural network using TensorFlow/Keras.

5. Use the following architecture as the starting point:

   Input
      ↓
   Dense(16, ReLU)
      ↓
   Dense(8, ReLU)
      ↓
   Dense(1)

6. Compile the model using:

   - Adam optimizer
   - Mean Squared Error (MSE) as loss
   - Mean Absolute Error (MAE) as metric

7. Train the model for 100 epochs and use part of the training data
   as validation data.

8. Plot:
   - Training Loss vs Validation Loss
   - Training MAE vs Validation MAE

9. Evaluate the model on the test set.

10. Calculate and display:
    - Test MSE
    - Test MAE

11. Create predictions for the test set and compare:

    Real Price vs Predicted Price

12. Create a scatter plot showing:
    - X axis → Real Price
    - Y axis → Predicted Price

13. Create a prediction for a new house.

14. Experiment with different neural network architectures using
    the build_model() function.

    Test at least:

    Network A:
    [4]

    Network B:
    [16, 8]

    Network C:
    [32, 16, 8]

15. Compare the architectures based on:
    - Number of parameters
    - Training performance
    - Validation performance
    - Test performance

Learning Goals:

- Understand how neural networks can be used for regression.
- Understand the difference between classification and regression.
- Understand why the output layer does not use sigmoid.
- Understand the role of MSE and MAE.
- Understand how a neural network predicts continuous numerical values.
- Practice evaluating regression models.
- Understand how model complexity affects performance.
- Identify possible underfitting and overfitting.
"""

# ============================================================
# Exercise 02 - Neural Network Regression: House Price Prediction
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Dataset
# ============================================================

# Features:
# [Surface Area, Bedrooms, Bathrooms, House Age, Distance from City Center]

X = np.array([
    [50, 1, 1, 30, 12],
    [60, 2, 1, 25, 10],
    [70, 2, 1, 20, 9],
    [80, 2, 1, 15, 8],
    [90, 3, 2, 12, 7],
    [100, 3, 2, 10, 6],
    [110, 3, 2, 8, 5],
    [120, 3, 2, 7, 5],
    [130, 4, 2, 5, 4],
    [140, 4, 2, 4, 4],

    [55, 1, 1, 35, 14],
    [65, 2, 1, 28, 11],
    [75, 2, 1, 22, 10],
    [85, 3, 1, 18, 9],
    [95, 3, 2, 15, 8],
    [105, 3, 2, 11, 7],
    [115, 3, 2, 9, 6],
    [125, 4, 2, 6, 5],
    [135, 4, 2, 5, 4],
    [145, 4, 3, 3, 3],

    [52, 1, 1, 40, 15],
    [68, 2, 1, 30, 12],
    [78, 2, 2, 25, 10],
    [88, 3, 2, 20, 9],
    [98, 3, 2, 14, 7],
    [108, 3, 2, 10, 6],
    [118, 4, 2, 8, 5],
    [128, 4, 2, 6, 4],
    [138, 4, 3, 4, 3],
    [150, 4, 3, 2, 2]
])

# Target:
# House Price (€)

y = np.array([
    95000, 115000, 135000, 155000, 180000,
    205000, 230000, 250000, 280000, 310000,

    90000, 120000, 140000, 165000, 190000,
    215000, 240000, 270000, 300000, 335000,

    85000, 110000, 145000, 170000, 195000,
    225000, 255000, 285000, 320000, 360000
])


# ============================================================
# 2. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. Feature Scaling
# ============================================================

# Neural networks generally work better when input features
# have similar numerical scales.

X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# ============================================================
# 4. Target Scaling
# ============================================================

# House prices are large values such as 95000, 230000, 360000.
# We scale the target to make optimization easier.

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
)

y_test = y_scaler.transform(
    y_test.reshape(-1, 1)
)


# Keep the original test prices for the final comparison.
y_test_original = y_scaler.inverse_transform(y_test)


# ============================================================
# 5. Build Neural Network
# ============================================================

def build_model(hidden_layers):
    """
    Build and compile a neural network for regression.

    Parameters:
        hidden_layers: list containing the number of neurons
                       for each hidden layer.

    Example:
        [16, 8]

    Architecture:
        Input -> Dense(16) -> Dense(8) -> Dense(1)
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
    #
    # No activation function is used because this is a regression
    # problem and the output can be any real number.
    model.add(
        tf.keras.layers.Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 6. Define Architectures
# ============================================================

architectures = {
    "Network A": [4],
    "Network B": [16, 8],
    "Network C": [32, 16, 8]
}


# ============================================================
# 7. Train All Networks
# ============================================================

results = {}
models = {}
histories = {}


for name, architecture in architectures.items():

    print("\n" + "=" * 50)
    print(name)
    print(f"Architecture: {architecture}")
    print("=" * 50)

    # Build model
    model = build_model(architecture)

    # Display architecture
    model.summary()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_split=0.2,
        verbose=0
    )

    # Save model and history
    models[name] = model
    histories[name] = history

    # ========================================================
    # Evaluate on test set
    # ========================================================

    test_loss, test_mae_scaled = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # ========================================================
    # Predictions
    # ========================================================

    y_pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    # Convert predictions back to euros
    y_pred = y_scaler.inverse_transform(
        y_pred_scaled
    )

    # Calculate metrics in original scale
    test_mae_euro = mean_absolute_error(
        y_test_original,
        y_pred
    )

    test_mse_euro = mean_squared_error(
        y_test_original,
        y_pred
    )

    # Training metrics
    training_mae_scaled = history.history["mae"][-1]
    validation_mae_scaled = history.history["val_mae"][-1]

    training_loss = history.history["loss"][-1]
    validation_loss = history.history["val_loss"][-1]

    # Save results
    results[name] = {
        "architecture": architecture,
        "parameters": model.count_params(),
        "training_mae_scaled": training_mae_scaled,
        "validation_mae_scaled": validation_mae_scaled,
        "test_mae_scaled": test_mae_scaled,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "test_loss_scaled": test_loss,
        "test_mae_euro": test_mae_euro,
        "test_mse_euro": test_mse_euro,
        "predictions": y_pred
    }

    # ========================================================
    # Print results
    # ========================================================

    print("\nResults:")

    print(
        f"Training MAE (scaled): "
        f"{training_mae_scaled:.4f}"
    )

    print(
        f"Validation MAE (scaled): "
        f"{validation_mae_scaled:.4f}"
    )

    print(
        f"Test MAE (scaled): "
        f"{test_mae_scaled:.4f}"
    )

    print(
        f"Training Loss: "
        f"{training_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Test Loss (scaled): "
        f"{test_loss:.4f}"
    )

    print(
        f"Test MAE: "
        f"{test_mae_euro:,.2f} €"
    )

    print(
        f"Test MSE: "
        f"{test_mse_euro:,.2f} €²"
    )


# ============================================================
# 8. Compare Networks
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK COMPARISON")
print("=" * 70)

print(
    f"{'Network':<12}"
    f"{'Architecture':<18}"
    f"{'Params':<10}"
    f"{'Test MAE':<15}"
    f"{'Test MSE':<15}"
)

print("-" * 70)

for name, result in results.items():

    print(
        f"{name:<12}"
        f"{str(result['architecture']):<18}"
        f"{result['parameters']:<10}"
        f"{result['test_mae_euro']:>10,.2f} €   "
        f"{result['test_mse_euro']:>12,.2f}"
    )


# ============================================================
# 9. Find Best Network
# ============================================================

# We select the network with the lowest test MAE.

best_network_name = min(
    results,
    key=lambda name: results[name]["test_mae_euro"]
)

best_model = models[best_network_name]
best_predictions = results[best_network_name]["predictions"]


print("\n")
print("=" * 70)
print("BEST NETWORK ON THIS TEST SPLIT")
print("=" * 70)

print(
    f"Network: {best_network_name}"
)

print(
    f"Architecture: "
    f"{results[best_network_name]['architecture']}"
)

print(
    f"Parameters: "
    f"{results[best_network_name]['parameters']}"
)

print(
    f"Test MAE: "
    f"{results[best_network_name]['test_mae_euro']:,.2f} €"
)

print(
    f"Test MSE: "
    f"{results[best_network_name]['test_mse_euro']:,.2f} €²"
)


# ============================================================
# 10. Real Price vs Predicted Price
# ============================================================

print("\n")
print("=" * 70)
print(
    f"REAL PRICE VS PREDICTED PRICE - {best_network_name}"
)
print("=" * 70)

for real, predicted in zip(
    y_test_original.flatten(),
    best_predictions.flatten()
):

    print(
        f"Real: {real:,.2f} € | "
        f"Predicted: {predicted:,.2f} €"
    )


# ============================================================
# 11. Plot Real vs Predicted
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_original,
    best_predictions
)

# Perfect prediction line
min_price = min(
    y_test_original.min(),
    best_predictions.min()
)

max_price = max(
    y_test_original.max(),
    best_predictions.max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price]
)

plt.xlabel("Real Price (€)")
plt.ylabel("Predicted Price (€)")

plt.title(
    f"Real vs Predicted Prices - {best_network_name}"
)

plt.grid(True)

plt.show()


# ============================================================
# 12. New House Prediction
# ============================================================

# New house:
#
# Surface Area: 115 m²
# Bedrooms: 3
# Bathrooms: 2
# House Age: 8 years
# Distance from City Center: 5 km

new_house = np.array([
    [115, 3, 2, 8, 5]
])


# Scale using the SAME scaler used during training.

new_house_scaled = X_scaler.transform(
    new_house
)


print("\n")
print("=" * 70)
print("NEW HOUSE PREDICTIONS")
print("=" * 70)

print(
    "House: 115 m² | "
    "3 bedrooms | "
    "2 bathrooms | "
    "8 years | "
    "5 km from city center"
)


# ============================================================
# 13. Prediction with Every Network
# ============================================================

for name, model in models.items():

    predicted_price_scaled = model.predict(
        new_house_scaled,
        verbose=0
    )

    # Convert prediction back to euros.
    predicted_price = y_scaler.inverse_transform(
        predicted_price_scaled
    )

    print(
        f"{name:<12}: "
        f"{predicted_price[0][0]:,.2f} €"
    )


# ============================================================
# 14. Training Curves - Best Network
# ============================================================

best_history = histories[best_network_name]


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    best_history.history["mae"],
    label="Training MAE"
)

plt.plot(
    best_history.history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")

plt.title(
    f"Training vs Validation MAE - {best_network_name}"
)

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    best_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    best_history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title(
    f"Training vs Validation Loss - {best_network_name}"
)

plt.legend()
plt.grid(True)

plt.show()

