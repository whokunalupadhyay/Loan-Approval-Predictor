import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pickle

# --- 1. Load the dataset ---

df = pd.read_excel('project_dataset.xlsx')

# --- Initial Data Cleaning 
print("Data summary\n")
print(df.isnull().sum())
print(df.info())
print(df.describe())
print("top 5 rows\n")
print(df.head())
print("checking how many duplicates are there")
duplicate = df.duplicated().sum()
print(duplicate)

# Drop rows where target (LoanApproved) is missing
df = df.dropna(subset=["LoanApproved"])
# Removing invalid labels (to keep only 0 and 1)
df = df[df["LoanApproved"].isin([0,1])].copy()
# Ensuring target is integer
df["LoanApproved"] = df["LoanApproved"].astype(int)

# --- 2. Define Features and Target ---

# Loan_ID is often dropped as it is a unique identifier
if 'Loan_ID' in df.columns:
    df = df.drop("Loan_ID", axis=1)

#Encoding categorical data
categorical_features = ['Education','SelfEmployed']
# Check which of these are actually present in the DataFrame before proceeding
categorical_features = [col for col in categorical_features if col in df.columns]

# Numerical columns 
numerical_features = ['ApplicantIncome','LoanAmount','CreditScore']
numerical_features = [col for col in numerical_features if col in df.columns]


# --- 4. Preprocessing Pipelines using ColumnTransformer ---

# Pipeline for Numerical Features (Impute missing data with mean, then Scale)
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Features (Impute missing categories with most frequent, then One-Hot Encode)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)

# --- 5. Split Data  ---
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 6. Creating Full Model Pipelines  ---

# Core correction: class_weight='balanced' addresses the root cause of majority-class prediction
log_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced'))
])
# Decision Tree Pipeline (keep it, but rename for clarity if desired)
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42)) # Added random_state
])


rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,             
        class_weight='balanced',      
        random_state=42, 
        n_jobs=-1                     
    ))
])


# --- 7. Train and Predict using the Pipelines ---

# Training Logistic Regression

log_pipeline.fit(X_train, y_train)
log_preds = log_pipeline.predict(X_test)

# Training Decision Tree
tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_test)

# Training Random Forest
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

# --- 8. Evaluation 
print("--- Evaluation Results ---")
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds)) 

# Classification Report 
print("\nClassification Report - Logistic Regression (Pay attention to Recall for '0'):\n", classification_report(y_test, log_preds, zero_division=1))
print("\nClassification Report - Decision Tree (Pay attention to Recall for '0'):\n", classification_report(y_test, tree_preds, zero_division=1))
print("\nClassification Report - Random Forest (Check Recall for '0'):\n", 
      classification_report(y_test, rf_preds, zero_division=1))

# ROC/AUC for Random Forest
rf_y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
print("AUC Score (Random Forest):", roc_auc_score(y_test, rf_y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, log_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_pipeline.named_steps['classifier'].classes_)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression (Balanced)")
plt.show()

# ROC Curve and AUC
rf_y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
y_prob = log_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Balanced)")
plt.grid()
plt.show()

print("AUC Score (Random Forest):", roc_auc_score(y_test, rf_y_prob))
print("AUC Score (Logistic Regression):", roc_auc_score(y_test, y_prob))

log_f1 = f1_score(y_test, log_preds)
rf_f1 = f1_score(y_test, rf_preds)
print(f"Logistic Regression F1-Score: {log_f1:.4f}")
print(f"Random Forest F1-Score: {rf_f1:.4f}")


# --- 9. Geting Feature Names for Deployment ---

preprocessor = rf_pipeline.named_steps['preprocessor']

# Geting all final feature names (numeric + one-hot encoded)
final_feature_names = preprocessor.get_feature_names_out()

# Clean up the names (e.g., remove 'num__', 'cat__')
cleaned_feature_names = [name.split('__')[-1] for name in final_feature_names]

# --- 10. Export  ---
export_object = {
    'model': rf_pipeline,
    'features': cleaned_feature_names
}

with open('loan_prediction_pipeline.pkl', 'wb') as file:
   
    pickle.dump(export_object, file)

print("\nRandom Forest Pipeline and Feature List exported successfully.")




