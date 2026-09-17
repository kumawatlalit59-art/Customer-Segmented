# ==========================================
# Customer Segmentation and Prediction Project
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("Customer_Segmentation_Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# ==========================
# Data Cleaning
# ==========================

print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================

le = LabelEncoder()

categorical_columns = [
    "Gender",
    "City",
    "State",
    "PreferredCategory"
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# ==========================
# Feature Engineering
# ==========================

df["CustomerLifetimeValue"] = (
    df["TotalPurchases"] *
    df["AverageOrderValue"]
)

# ==========================
# Customer Segmentation
# ==========================

features = [
    "Income",
    "PurchaseFrequency",
    "AverageOrderValue",
    "CustomerLifetimeValue"
]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[features])

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_data)

print("\nCluster Counts")
print(df["Cluster"].value_counts())

print("\nCluster Summary")
print(df.groupby("Cluster").mean(numeric_only=True))

# ==========================
# Scatter Plot
# ==========================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Income",
    y="CustomerLifetimeValue",
    hue="Cluster",
    palette="Set2",
    s=80
)

plt.title("Customer Segmentation")
plt.xlabel("Income")
plt.ylabel("Customer Lifetime Value")

plt.show()

# ==========================
# Prediction Model
# ==========================

# IMPORTANT:
# CustomerID and CustomerName are NOT included.

X = df[
[
    "Gender",
    "Age",
    "City",
    "State",
    "Income",
    "TotalPurchases",
    "PurchaseFrequency",
    "AverageOrderValue",
    "LastPurchaseDays",
    "CustomerLifetimeValue",
    "Cluster"
]
]

y = df["FuturePurchase"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("RMSE :", rmse)
print("R2 Score :", r2)

# ==========================
# Actual vs Predicted
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    pred
)

plt.xlabel("Actual Future Purchase")
plt.ylabel("Predicted Future Purchase")
plt.title("Actual vs Predicted")

plt.show()

# ==========================
# Purchase Frequency
# ==========================

plt.figure(figsize=(8,6))

sns.histplot(
    df["PurchaseFrequency"],
    bins=20,
    kde=True,
    color="green"
)

plt.title("Purchase Frequency Distribution")

plt.show()

# ==========================
# Income Distribution
# ==========================

plt.figure(figsize=(8,6))

sns.boxplot(
    y=df["Income"],
    color="orange"
)

plt.title("Income Distribution")

plt.show()

# ==========================
# Correlation Heatmap
# ==========================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================
# Customer Count by Cluster
# ==========================

plt.figure(figsize=(8,6))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Cluster",
    palette="Set2",
    legend=False
)

plt.title("Customers in Each Cluster")

plt.show()

# ==========================
# Average CLV by Cluster
# ==========================

plt.figure(figsize=(8,6))

sns.barplot(
    data=df,
    x="Cluster",
    y="CustomerLifetimeValue",
    estimator=np.mean,
    palette="Set2"
)

plt.title("Average Customer Lifetime Value by Cluster")

plt.show()

# ==========================
# Save Output
# ==========================

df.to_csv(
    "Customer_Segmentation_Output.csv",
    index=False
)

print("\nProject Completed Successfully!")
print("Output saved as Customer_Segmentation_Output.csv")