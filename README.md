# 💰 Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A complete **Machine Learning workflow** for predicting whether a loan application will be **approved (1)** or **rejected (0)** based on applicant financial and demographic data.  
The project includes **data preprocessing, model training, evaluation, and deployment** as a **Streamlit web app**.

---

## 🚀 Features

✅ **Data Pipeline:**  
Built using `scikit-learn`’s **Pipelines** and **ColumnTransformer** for clean, modular preprocessing.  
- Numerical: Scaling + Median Imputation  
- Categorical: One-Hot Encoding + Most Frequent Imputation  

✅ **Model Comparison:**  
Trains & compares:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
Evaluated on **F1-Score**, **AUC**, and **Confusion Matrix**.

✅ **Balanced Classes:**  
Implements `class_weight='balanced'` to improve recall on minority classes.

✅ **Interactive Web App (Streamlit):**  
User-friendly interface for:
- 🔮 Real-time loan predictions  
- 📊 Simple dashboard tracking input history & insights  

✅ **Feature Influence:**  
Displays model coefficients or feature importance for interpretability.

---

## 🧩 Project Structure

| File | Description |
|------|--------------|
| `project_dataset.xlsx` | Dataset containing applicant & loan details |
| `train_model.py` | Trains models, builds pipelines & saves the best one |
| `app.py` | Streamlit web application |
| `loan_prediction_pipeline.pkl` | Serialized ML pipeline (generated after training) |

---

## ⚙️ Setup and Installation

### 🧱 Prerequisites
Ensure you have **Python 3.8+** installed.
```bash
python --version
```

## 📦 Install Dependencies

### Create a virtual environment (recommended) and install the required packages:
```bash
pip install -r requirements.txt
```



##  🧭 How to Run the Entire Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Train the model
```bash
python train_model.py
```
#### 4️⃣ Run the web app
```bash
streamlit run app.py
```
