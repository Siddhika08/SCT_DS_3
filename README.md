# SCT_DS_3
Build a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing Dataset from the UCI Machine Learning Repository.

## 🎯 Objective
The goal of this project is to implement a **supervised machine learning model** — a **Decision Tree Classifier** — that can predict customer behavior based on real-world marketing data.
## ⚙️ Steps Involved

### 1️⃣ Data Loading
Imported the Bank Marketing dataset directly from the UCI Machine Learning Repository.

### 2️⃣ Data Cleaning & Preprocessing
- Checked for missing values  
- Encoded categorical variables using **Label Encoding**

### 3️⃣ Feature & Target Selection
- Defined input features (`X`) and output target (`y`)

### 4️⃣ Train-Test Split
- Split dataset into 80% training and 20% testing sets

### 5️⃣ Model Building
- Used **DecisionTreeClassifier** from Scikit-learn  
- Trained the model on the training data

### 6️⃣ Model Evaluation
- Evaluated model performance using:
  - Accuracy
  - Confusion Matrix
  - Classification Report  
- Visualized Decision Tree structure
## 🧠 Technologies & Libraries Used
- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
## 📊 Results
- The Decision Tree model successfully predicts whether a customer will subscribe to a service.  
- Achieved good accuracy and visualized decision paths.  
## 🔍 Dataset Used
📂 **Bank Marketing Dataset**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
