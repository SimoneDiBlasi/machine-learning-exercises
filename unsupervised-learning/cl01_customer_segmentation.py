"""
CL01 — Customer Segmentation with K-Means

EXERCISE DESCRIPTION

You are a data analyst working for an e-commerce company. The company
wants to segment its customers into groups based on their purchasing
behavior, but no predefined customer categories are available.

The dataset contains information about each customer, including:

- Annual Income (k$): The customer's annual income in thousands of dollars.
- Spending Score (1-100): A score assigned to each customer based on
  their spending behavior and purchasing habits.

Your goal is to apply the K-Means clustering algorithm to identify
groups of customers with similar characteristics.


REQUIREMENTS

1. Load the dataset using Pandas.
2. Explore the dataset and analyze the distribution of its features.
3. Create a scatter plot to visualize the relationship between annual
   income and spending score.
4. Apply feature scaling using StandardScaler.
5. Use the Elbow Method to determine an appropriate number of clusters.
6. Implement the K-Means algorithm using the selected number of clusters.
7. Assign a cluster label to each customer.
8. Visualize the resulting clusters using different colors and display
   their centroids.
9. Analyze the characteristics of each cluster and describe the
   customer segments identified.
10. Calculate the Silhouette Score to evaluate the quality of the clustering.


CONSTRAINTS

- Do not use predefined target labels to train the model.
- Do not assume the optimal number of clusters without justification.
- Use appropriate visualizations to support your analysis.
- Explain the reasoning behind the main steps of your implementation.


DATASET

Use the Mall Customer Segmentation Dataset, available on Kaggle.

The dataset typically includes customer ID, gender, age, annual income,
and spending score.

For this exercise, focus on the following features:
- Annual Income (k$)
- Spending Score (1-100)


QUESTIONS TO ANSWER

1. What is clustering, and why is K-Means considered an unsupervised
   learning algorithm?
2. Why is feature scaling important when applying K-Means?
3. How does the Elbow Method help determine the number of clusters?
4. What does a centroid represent in K-Means?
5. How can you interpret the customer segments identified by the algorithm?
6. What does the Silhouette Score measure, and how can it help evaluate
   clustering quality?
7. What are the main limitations of K-Means?


EXPECTED OUTCOME

At the end of the exercise, you should have a Python script that loads
and explores the dataset, applies K-Means clustering, evaluates the
results, and visualizes the identified customer segments.

You should also be able to explain how the algorithm works and why the
selected number of clusters is appropriate.
"""

import kagglehub
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Download latest version
path :str = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
dataset_path = next(Path(path).glob("*.csv"))
df = pd.read_csv(dataset_path)
print("First five rows:")
print(df.head())

# Display dataset information
print("\nDataset information:")
df.info()

# Display statistical summaries
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Display the shape of the dataset
print("\nDataset shape:", X.shape)


# Visualize the customers
plt.figure(figsize=(8, 6))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    alpha=0.7
)

plt.title("Customer Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Display the first five scaled rows
print("\nFirst five rows after scaling:")
print(X_scaled[:5])

# Display the mean and standard deviation of the scaled features
print("\nFeature means:")
print(X_scaled.mean(axis=0))

print("\nFeature standard deviations:")
print(X_scaled.std(axis=0))

inertia_values = []

k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df["Cluster"] = cluster_labels

# Display the number of customers in each cluster
print("\nCustomers per cluster:")
print(df["Cluster"].value_counts().sort_index())

# Display the cluster centroids in the scaled feature space
print("\nScaled cluster centroids:")
print(kmeans.cluster_centers_)

# Analyze the average characteristics of each cluster
cluster_summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean().round(2)

print("\nCluster summary:")
print(cluster_summary)

# Calculate the Silhouette Score
score = silhouette_score(X_scaled, cluster_labels)

print(f"\nSilhouette Score: {score:.4f}")


# Compare different numbers of clusters
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")

plt.title("Silhouette Score for Different Values of K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find the K with the highest Silhouette Score
best_index = silhouette_scores.index(max(silhouette_scores))
best_k = list(k_values)[best_index]

print(f"\nBest K according to Silhouette Score: {best_k}")
print(f"Highest Silhouette Score: {silhouette_scores[best_index]:.4f}")

print("\n" + "=" * 60)
print("CUSTOMER CLUSTER SEGMENTATION")
print("=" * 60)

for cluster_id, row in cluster_summary.iterrows():
    customers_count = (df["Cluster"] == cluster_id).sum()

    print(f"\n{'-' * 60}")
    print(f"CLUSTER {cluster_id}")
    print("-" * 60)
    print(f"Number of customers: {customers_count}")
    print(f"Average annual income: ${row['Annual Income (k$)']:.2f}k")
    print(f"Average spending score: {row['Spending Score (1-100)']:.2f}")