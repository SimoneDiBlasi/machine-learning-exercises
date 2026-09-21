"""
Exercise 03 - Logistic Regression: Customer Purchase Prediction

Objective:
    Use Logistic Regression to predict whether a customer
    will purchase a product based on multiple features.

Features:
    - Age
    - Salary

Target:
    - 0 = No purchase
    - 1 = Purchase

Tasks:
    1. Create the dataset
    2. Split the dataset into training and test sets
    3. Create a Logistic Regression model
    4. Train the model
    5. Inspect the model coefficients and intercept
    6. Predict whether a new customer will purchase the product
    7. Predict the classes for the test set
    8. Print the real and predicted results
    9. Visualize the real and predicted classes

Concepts:
    - Classification
    - Train/Test Split
    - Logistic Regression
    - Model Training
    - Prediction
    - Coefficients
    - Intercept
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Dataset
# X contains two input features:
# 1. Age
# 2. Annual salary
X = np.array([
    [20, 20000],
    [22, 25000],
    [25, 30000],
    [28, 35000],
    [30, 40000],
    [35, 45000],
    [40, 50000],
    [45, 60000],
    [50, 70000],
    [55, 80000],
    [60, 90000],
    [65, 100000]
])

# y contains the target:
# 0 = No purchase
# 1 = Purchase
y = np.array([
    0,
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
    1
])


# Train / Test Split
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


# Model
model = LogisticRegression()

# Training
model.fit(X_train, y_train)


# Model Parameters
coefficients = model.coef_
intercept = model.intercept_

print("\nCoefficients:", coefficients)
print("Intercept:", intercept)


# Prediction for a New Customer
new_customer = np.array([[18, 15000]])

purchase_prediction = model.predict(new_customer)

print("\nNew customer prediction:")
print("Will purchase:", purchase_prediction[0])

purchase_probability = model.predict_proba(new_customer)

print("\nPurchase probabilities:")
print("No purchase:", purchase_probability[0][0])
print("Purchase:", purchase_probability[0][1]) 
# Prediction on the Test Set
y_pred = model.predict(X_test)

print("\nReal purchase results:")
print(y_test)

print("\nPredicted purchase results:")
print(y_pred)


# Compare Real and Predicted Results
print("\nReal vs Predicted:")

for real, predicted in zip(y_test, y_pred):
    print(f"Real: {real} | Predicted: {predicted}")


# Visualization
plt.scatter(
    range(len(y_test)),
    y_test,
    label="Real"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Purchase (0 = No, 1 = Yes)")
plt.title("Logistic Regression - Real vs Predicted")
plt.legend()
plt.show()