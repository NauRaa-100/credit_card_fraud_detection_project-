
#  Credit Card Fraud Detection

A complete **Machine Learning project** for detecting fraudulent transactions using the **Credit Card Fraud Detection Dataset**.  
This project demonstrates the full ML workflow — from **data cleaning, optimization, visualization, feature selection, and model evaluation** — all the way to **model saving** for deployment.

---

##  Overview

- **Goal:** Detect fraudulent transactions (Class = 1) vs. legitimate ones (Class = 0).  
- **Dataset:** Credit Card Fraud Detection (highly imbalanced dataset).  
- **ML Type:** Binary Classification  
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn

---

##  Steps & Workflow

### 1. Data Loading & Overview
- Loaded CSV dataset using `pandas`.
- Displayed shape, info, data types, missing values.
- Optimized data types to reduce memory usage (e.g., `float64` → `float32`, `int64` → `int8`).

### 2. Outlier Detection & Removal
- Outliers detected using the **IQR (Interquartile Range)** method.  
- Removed sensitive outliers from `Amount` and `Time` columns.

### 3. Exploratory Data Analysis (EDA)
- Checked **class distribution** and correlation with `Class`.  
- Visualized with **Seaborn** (`countplot`, `heatmap`, etc.).

### 4. Data Scaling
- Scaled numeric features using **RobustScaler** to reduce the effect of outliers.

### 5. Handling Imbalance
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic samples and balance the dataset.

### 6. Feature Selection
- Trained a **RandomForestClassifier** to extract **feature importances**.  
- Selected top 10 important features.

### 7. Model Training & Evaluation
Trained multiple models:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Ridge Classifier  
- Gradient Boosting  
- Decision Tree  
- Perceptron  

Evaluated each using:
- **F1-score**
- **Confusion Matrix**
- **ROC Curve** and **AUC Score**

**Best model:** Logistic Regression 

---

##  Visualizations

- **Class Distribution Plot**
- **Confusion Matrix (Heatmap)**
- **ROC Curve with AUC Score**
- **Feature Correlation Heatmap**

---

##  Saving the Model

Saved the best model and scaler for deployment using `joblib`:

```python
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'best_model_.pkl')
````

---

##  Technologies Used

| Category          | Libraries                |
| ----------------- | ------------------------ |
| Data Manipulation | pandas, numpy            |
| Visualization     | seaborn, matplotlib      |
| ML Algorithms     | scikit-learn             |
| Imbalanced Data   | imbalanced-learn (SMOTE) |
| Model Saving      | joblib                   |

---

##  Results

| Metric            | Value               |
| ----------------- | ------------------- |
| Best Model        | Logistic Regression |
| Evaluation Metric | F1 Score            |
| Balanced Dataset  | Yes (SMOTE applied) |
| Visual AUC Score  | ~0.99               |

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://https://github.com/NauRaa-100/credit_card_fraud_detection_project-/blob/main/credit.py
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python fraud_detection.py
   ```

---

##  Project Structure

```
 CreditCard-Fraud-Detection
├── fraud_detection.py
├── creditcard.csv (big size -- You can Download from Kaggle)
├── scaler.pkl
├── best_model_.pkl
├── README.md
└── requirements.txt
```

---

##  Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
imbalanced-learn
joblib
```

---

##  Key Insights

* Data is **highly imbalanced** — handled using SMOTE.
* **RobustScaler** performed best due to outlier resistance.
* **Logistic Regression** achieved the most stable performance across metrics.
* Visualizations confirmed strong model separation between classes.

---

## Author

**Nau Raa**
Data Science & AI Enthusiast
2025

---

*If you like this project, don't forget to give it a star on GitHub!*


