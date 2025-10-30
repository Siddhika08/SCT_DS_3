# ---------------------------------------------------------------
# TASK 03: Decision Tree Classifier - Bank Marketing Dataset
# ---------------------------------------------------------------
# 🎯 Objective:
# Build a Decision Tree Classifier to predict whether a customer 
# will purchase a product or service based on their demographic 
# and behavioral data.
# ---------------------------------------------------------------

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
# You can download the dataset from UCI ML Repository
# or load a CSV file if you have it locally
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv"
data = pd.read_csv(url, sep=';')

# Step 3: Display first few rows
print("🔹 First 5 rows of dataset:")
print(data.head())

# Step 4: Basic dataset info
print("\n📊 Dataset Info:")
print(data.info())

# Step 5: Check for missing values
print("\n🧩 Missing Values in each column:")
print(data.isnull().sum())

# Step 6: Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Step 7: Define features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Step 8: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Create and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate the model
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Step 12: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 13: Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
