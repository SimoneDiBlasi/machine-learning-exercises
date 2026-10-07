"""
Exercise 04 - Logistic Regression: Customer Churn Prediction

Objective:
    Use Logistic Regression to predict whether a customer
    will leave a company based on multiple features.

Features:
    - Age
    - Monthly charge
    - Months with company

Target:
    - 0 = No churn
    - 1 = Churn

Tasks:
    1. Create the dataset
    2. Split the dataset into training and test sets
    3. Create a Logistic Regression model
    4. Train the model
    5. Inspect the model coefficients and intercept
    6. Predict whether a new customer will churn
    7. Calculate the churn probabilities
    8. Predict the classes for the test set
    9. Calculate the model accuracy
    10. Compare the real and predicted results
    11. Visualize the real and predicted classes

Concepts:
    - Classification
    - Train/Test Split
    - Logistic Regression
    - Model Training
    - Prediction
    - Prediction Probability
    - Accuracy
    - Coefficients
    - Intercept
"""


from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Dataset
# X contains three input features:
# 1. Age
# 2. Monthly charge
# 3. Months with company
X = np.array([
    [22, 30, 3],
    [25, 35, 6],
    [28, 40, 12],
    [30, 45, 24],
    [35, 50, 36],
    [40, 60, 48],
    [45, 70, 60],
    [50, 80, 72],
    [55, 90, 84],
    [60, 100, 96],
    [65, 110, 108],
    [70, 120, 120]
])


# y contains the target:
# 0 = No churn
# 1 = Churn
y = np.array([
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
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
new_customer = np.array([[28, 20, 2]])

churn_prediction = model.predict(new_customer)
churn_probability = model.predict_proba(new_customer)

print("\nNew customer prediction:")
print("Will churn:", churn_prediction[0])

print("\nChurn probabilities:")
print("No churn:", churn_probability[0][0])
print("Churn:", churn_probability[0][1])


# Prediction on the Test Set
y_pred = model.predict(X_test)


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


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
plt.ylabel("Churn (0 = No, 1 = Yes)")
plt.title("Logistic Regression - Real vs Predicted")
plt.legend()
plt.show()