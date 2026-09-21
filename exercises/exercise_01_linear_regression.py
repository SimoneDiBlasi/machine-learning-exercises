"""
Exercise 01 - Linear Regression: House Price Prediction

Objective:
    Use Linear Regression to predict the price of a house
    based on its surface area.

Concepts:
    - Train/Test Split
    - Linear Regression
    - Model Training
    - Prediction
    - Coefficient and Intercept
    - Mean Squared Error (MSE)
    - R² Score
    - Regression Line Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Dataset
# ============================================================

# X = input feature → house surface area in square meters
X = np.array([
    [50],
    [60],
    [70],
    [80],
    [90],
    [100],
    [120],
    [150]
])

# y = target → house price in euros
y = np.array([
    110000,
    125000,
    140000,
    155000,
    175000,
    195000,
    230000,
    290000
])


# ============================================================
# Train / Test Split
# ============================================================

# Split the dataset into:
# - 80% training data
# - 20% test data
#
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Model
# ============================================================

# Create a Linear Regression model.
model = LinearRegression()


# ============================================================
# Training
# ============================================================

# Train the model using the training data.
# The model learns the relationship between
# house surface area and price.
model.fit(X_train, y_train)


# ============================================================
# Model Parameters
# ============================================================

# Coefficient:
# Represents the expected change in the predicted price
# for each additional square meter.
coefficient = model.coef_[0]

# Intercept:
# Represents the predicted price when the input feature is 0.
intercept = model.intercept_

print("Coefficient:", coefficient)
print("Intercept:", intercept)


# ============================================================
# Prediction for a New House
# ============================================================

# Predict the price of a new house with an area of 110 m².
house = np.array([[110]])

house_prediction = model.predict(house)

print("\nPredicted price for 110 m²:", house_prediction[0])


# ============================================================
# Prediction on the Test Set
# ============================================================

# Use the trained model to predict prices for the test set.
y_pred = model.predict(X_test)

print("\nReal prices:")
print(y_test)

print("\nPredicted prices:")
print(y_pred)


# ============================================================
# Model Evaluation
# ============================================================

# Mean Squared Error (MSE):
# Calculates the average squared difference between
# the actual and predicted values.
mse = mean_squared_error(y_test, y_pred)

# R² Score:
# Measures how well the model explains the variance
# in the target values.
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R²:", r2)


# ============================================================
# Visualization
# ============================================================

# Plot the actual data points.
plt.scatter(X, y)

# Plot the Linear Regression line.
plt.plot(X, model.predict(X))

plt.xlabel("Surface Area (m²)")
plt.ylabel("Price (€)")
plt.title("Linear Regression - House Prices")

plt.show()