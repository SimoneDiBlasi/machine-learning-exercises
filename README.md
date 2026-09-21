# Machine Learning Exercises

This repository contains practical Python exercises to study and practice Machine Learning.

The goal is to build a solid understanding of Machine Learning concepts through progressively more complex exercises.

---

## Project Setup

The project is developed in Python using a virtual environment.

### 1. Create the project folder

The project was created with the following structure:

```text
machine-learning-exercises/
```

### 2. Create the virtual environment

A dedicated Python virtual environment was created for the entire Machine Learning project:

```powershell
python -m venv .venv
```

The virtual environment is stored in:

```text
.venv/
```

The virtual environment is shared by all exercises in this project.

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

When the environment is active, the terminal shows:

```text
(.venv)
```

This means that Python and the installed packages are being used from the project's virtual environment.

### 4. Install the required libraries

The main libraries used by the project are:

- NumPy — numerical operations and arrays
- Pandas — data manipulation and datasets
- Matplotlib — data visualization
- Scikit-learn — Machine Learning algorithms and metrics

They were installed with:

```powershell
pip install numpy pandas matplotlib scikit-learn
```

### 5. Save the project dependencies

The installed packages were saved to:

```text
requirements.txt
```

using:

```powershell
pip freeze > requirements.txt
```

This allows the environment to be recreated in the future with:

```powershell
pip install -r requirements.txt
```

---

## Project Structure

The current project structure is:

```text
machine-learning-exercises/
│
├── .venv/
│
├── exercises/
│   ├── exercise_01_linear_regression.py
│   └── exercise_02_multiple_linear_regression.py
│
├── requirements.txt
│
└── README.md
```

### Folder and file description

#### `.venv/`

Python virtual environment containing the interpreter and project dependencies.

This folder should normally not be committed to Git.

#### `exercises/`

Contains the individual Machine Learning exercises.

Each exercise will be stored in a separate Python file.

#### `requirements.txt`

Contains the Python packages required by the project.

#### `README.md`

Contains the project documentation, setup instructions, topics studied and descriptions of the exercises.

---

# Machine Learning Topics

At the moment, the topics studied are:

```text
Supervised Learning
│
├── Regression
│   └── Linear Regression
│
└── Classification
    └── Logistic Regression
```

Only these topics have been studied so far.

Future topics will be added progressively as they are studied.

---

# Exercise 01 — Linear Regression

## Objective

Build a simple Machine Learning model using **Linear Regression** to predict the price of a house based on its surface area.

The exercise is intentionally simple because the objective is to practice the complete Machine Learning workflow:

```text
Dataset
   ↓
Define X and y
   ↓
Split the dataset
   ↓
Create the model
   ↓
Train the model
   ↓
Make predictions
   ↓
Evaluate the model
   ↓
Interpret the model
```

---

## Dataset

We have a small dataset containing the surface area of houses and their corresponding prices.

```python
import numpy as np

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
```

### Features

`X` represents the input feature:

```text
Surface area in square meters
```

### Target

`y` represents the value that we want the model to predict:

```text
House price
```

---

## Tasks

Complete the following tasks using `scikit-learn`.

### 1. Split the dataset

Use `train_test_split` with:

```text
test_size = 0.2
random_state = 42
```

### 2. Create the Linear Regression model

Create an instance of:

```python
LinearRegression()
```

### 3. Train the model

Train the model using:

```python
model.fit(...)
```

### 4. Predict the price of a 110 m² house

Create an input representing a house of 110 m² and use:

```python
model.predict(...)
```

### 5. Predict the prices of the test set

Store the predictions in:

```python
y_pred
```

Print the real and predicted prices.

### 6. Calculate the Mean Squared Error

Calculate the MSE using:

```python
mean_squared_error(...)
```

### 7. Calculate R²

Calculate the R² score using:

```python
r2_score(...)
```

### 8. Inspect the model parameters

Print:

```python
model.coef_
model.intercept_
```

Relate them to:

```text
y = mx + b
```

### 9. Visualize the regression

Create a scatter plot containing the original data points and the regression line using Matplotlib.

---

## Questions to answer

1. Why do we separate `X` and `y`?
2. Why do we split the dataset into training and test data?
3. What does `fit()` do?
4. What does `predict()` return?
5. What do `coef_` and `intercept_` represent?

---

## Exercise Goal

The objective is **not simply to obtain the correct prediction**.

The important part is understanding the complete Machine Learning workflow:

```text
1. Prepare the data
        ↓
2. Separate features and target
        ↓
3. Split training/test data
        ↓
4. Create the model
        ↓
5. Train the model
        ↓
6. Make predictions
        ↓
7. Evaluate the model
        ↓
8. Interpret the result
        ↓
9. Visualize the model
```

This workflow will be reused and expanded in the following exercises.

---

# Exercise 02 — Multiple Linear Regression

## Objective

Extend the Linear Regression workflow from Exercise 01 by using **multiple input features** to predict the price of a house.

Instead of using only the surface area, the model will use:

- Surface area
- Number of bedrooms
- House age

The objective is to understand how Linear Regression works when a prediction depends on more than one feature.

---

## Dataset

Each row of `X` contains three features:

```text
[area, bedrooms, age]
```

```python
import numpy as np

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
```

### Features

Each row represents one house:

```text
[area, bedrooms, age]
```

For example:

```text
[100, 3, 5]
```

represents a house with:

```text
100 m²
3 bedrooms
5 years old
```

### Target

`y` represents the house price in euros.

---

## Tasks

Complete the following tasks using `scikit-learn`.

### 1. Split the dataset

Use `train_test_split` with:

```text
test_size = 0.2
random_state = 42
```

### 2. Create the Linear Regression model

Create an instance of:

```python
LinearRegression()
```

### 3. Train the model

Train the model using the training data with:

```python
model.fit(...)
```

### 4. Inspect the model parameters

Print:

```python
model.coef_
model.intercept_
```

Pay attention to the fact that `model.coef_` now contains **one coefficient for each feature**.

### 5. Predict the price of a new house

Predict the price of a house with:

```text
Area: 120 m²
Bedrooms: 3
Age: 5 years
```

Represent the input as:

```python
[[120, 3, 5]]
```

Use `model.predict(...)` and print the predicted price.

### 6. Predict the prices of the test set

Store the predictions in:

```python
y_pred
```

Print the real and predicted prices.

### 7. Calculate the Mean Squared Error

Calculate the MSE using:

```python
mean_squared_error(...)
```

### 8. Calculate R²

Calculate the R² score using:

```python
r2_score(...)
```

---

## Questions to answer

Answer these questions in your own words.

### 1. What is the main difference between Exercise 01 and Exercise 02?

### 2. What does `model.coef_` contain in Exercise 02?

### 3. If the model has 3 input features, how many coefficients should it have?

### 4. What does each coefficient represent?

### 5. Why does `X` now have 3 columns?

### 6. What does one row of `X` represent?

---

## Exercise Goal

The main objective is to understand how the Linear Regression model changes when multiple features are used.

In Exercise 01:

```text
Surface area ───────────────► House price
```

In Exercise 02:

```text
Surface area ────────────────┐
Bedrooms ───────────────────┼──► House price
House age ──────────────────┘
```

The Machine Learning workflow remains the same:

```text
Prepare the data
      ↓
Separate features and target
      ↓
Split training/test data
      ↓
Create the model
      ↓
Train the model
      ↓
Make predictions
      ↓
Evaluate the model
      ↓
Interpret the model
```

The main new concept is that the model now learns a relationship between **multiple input features and one target**.
