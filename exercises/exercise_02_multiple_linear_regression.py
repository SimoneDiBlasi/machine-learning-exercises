"""
Exercise 02 - Multiple Linear Regression: House Price Prediction

Objective:
    Use Multiple Linear Regression to predict the price of a house
    based on multiple features.

Features:
    - Surface area
    - Number of bedrooms
    - House age

Concepts:
    - Multiple Input Features
    - Train/Test Split
    - Multiple Linear Regression
    - Model Training
    - Prediction
    - Coefficients
    - Intercept
    - Mean Squared Error (MSE)
    - R² Score
    - Real vs Predicted Visualization
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# Dataset
# ============================================================

# X contains three input features:
# 1. Surface area in square meters
# 2. Number of bedrooms
# 3. House age in years
X = np.array([
    [50, 1, 30],
    [60, 2, 20],
    [70, 2, 15],
    [80, 2, 10],
    [90, 3, 8],
    [100, 3, 5],
    [120, 3, 3],
    [150, 4, 1]
])

# y contains the target values: house prices in euros.
y = np.array([
    100000,
    130000,
    150000,
    175000,
    200000,
    225000,
    270000,
    350000
])


# ============================================================
# Train / Test Split
# ============================================================

# Split the dataset into training and test sets.
# 80% of the data will be used for training.
# 20% of the data will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set:")
print(X_train)

print("\nTest set:")
print(X_test)


# ============================================================
# Model
# ============================================================

# Create the Multiple Linear Regression model.
model = LinearRegression()


# ============================================================
# Training
# ============================================================

# Train the model using the training data.
model.fit(X_train, y_train)


# ============================================================
# Model Parameters
# ============================================================

# Get the coefficients of the model.
# There is one coefficient for each input feature.
coefficients = model.coef_

# Get the intercept of the model.
intercept = model.intercept_

print("\nCoefficients:", coefficients)
print("Intercept:", intercept)


# ============================================================
# Prediction for a New House
# ============================================================

# Create a new house:
# - 110 m²
# - 2 bedrooms
# - 15 years old
house = np.array([[110, 2, 15]])


# Predict the price of the new house.
house_prediction = model.predict(house)

print("\nPredicted price for the new house:", house_prediction[0])


# ============================================================
# Prediction on the Test Set
# ============================================================

# Predict the prices of the houses in the test set.
y_pred = model.predict(X_test)

print("\nReal prices:")
print(y_test)

print("\nPredicted prices:")
print(y_pred)


# ============================================================
# Model Evaluation
# ============================================================

# Calculate the Mean Squared Error (MSE).
# MSE measures the average squared difference between
# the real prices and the predicted prices.
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error (MSE):", mse)


# Calculate the R² score.
# R² measures how well the model explains the variance
# in the real target values.
r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)
for real, predicted in zip(y_test, y_pred):
    print(f"Real: €{real:.2f} | Predicted: €{predicted:.2f}")

# ============================================================
# Visualization
# ============================================================

# Visualize real prices vs predicted prices.
# Each point represents one house from the test set.
plt.scatter(y_test, y_pred)

plt.xlabel("Real Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Real vs Predicted House Prices")

# Ideal predictions: predicted price = real price
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()