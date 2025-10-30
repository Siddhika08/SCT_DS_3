# ====================================================================
# TASK 3: Decision Tree Classifier using Bank Marketing Dataset (HTML)
# ====================================================================
# 📌 Objective:
# Read dataset information directly from the UCI Repository HTML page,
# extract useful tables, and then (optionally) download and use the data.
# ====================================================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Define the UCI dataset HTML page URL
html_url = "https://archive.ics.uci.edu/dataset/222/bank+marketing"

# Step 3: Read all tables from the HTML page
print("📥 Reading tables from the UCI dataset page...\n")
tables = pd.read_html(html_url)

# Step 4: Display basic info about tables found
print(f"✅ Total tables found on the webpage: {len(tables)}")

# Preview each table (to understand what’s inside)
for i, table in enumerate(tables):
    print(f"\n📋 Table {i+1} preview:")
    print(table.head())
    print("-" * 60)

# Step 5: Try to read the actual dataset from CSV (if available online)
print("\n📂 Attempting to load the actual dataset CSV from UCI...")
try:
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional/bank-additional-full.csv"
    data = pd.read_csv(csv_url, sep=';')
    print("✅ CSV dataset successfully loaded!\n")
except Exception as e:
    print("⚠️ Could not load the CSV dataset automatically.")
    print("Error details:", e)
    print("\nUsing sample data instead for demonstration purposes.\n")

    # Create a small sample dataset if CSV isn't available
    data = pd.DataFrame({
        'age': [30, 40, 50, 35, 28],
        'job': ['admin', 'technician', 'blue-collar', 'admin', 'services'],
        'marital': ['married', 'single', 'married', 'single', 'married'],
        'education': ['secondary', 'tertiary', 'primary', 'secondary', 'secondary'],
        'balance': [1000, 1500, 500, 1200, 300],
        'y': ['yes', 'no', 'no', 'yes', 'no']
    })

# Step 6: Display the dataset overview
print("\n🔹 Dataset Preview:")
print(data.head())

# Step 7: Encode categorical columns into numeric values
print("\n🔧 Encoding categorical variables...")
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 8: Split dataset into features and target variable
X = data_encoded.drop('y_yes', axis=1)
y = data_encoded['y_yes']

# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 10: Initialize and train Decision Tree model
print("\n🌳 Training Decision Tree Classifier...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 11: Make predictions
y_pred = model.predict(X_test)

# Step 12: Evaluate the model
print("\n📈 Model Evaluation Results:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 13: Save one of the HTML tables for record
tables[0].to_csv("bank_marketing_description.csv", index=False)
print("\n💾 Saved description table as 'bank_marketing_description.csv'")

print("\n✅ Task 3 completed successfully!")

