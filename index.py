import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
df = pd.read_excel('project_dataset.xlsx')

#data summary
print("Data summary\n")
print(df.isnull().sum())
print(df.info())
print(df.describe())
#top 5 rows
print("top 5 rows\n")
print(df.head())
#last 5 rows
print("last 5 rows\n")
print(df.tail())
#checking for duplicates
print("checking how many duplicates are there")
duplicate=df.duplicated().sum()
print(duplicate)

# Drop rows where target (LoanApproved) is missing
df = df.dropna(subset=["LoanApproved"])

# Remove invalid labels (keep only 0 and 1)
df = df[df["LoanApproved"].isin([0, 1])]

# Handle missing numeric values with mean imputation
numeric_cols = ["ApplicantIncome", "LoanAmount", "CreditScore"]
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Encode categorical columns
label_enc = LabelEncoder()
df["Education"] = label_enc.fit_transform(df["Education"])
df["SelfEmployed"] = label_enc.fit_transform(df["SelfEmployed"])

# Split into features and target
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"].astype(int)

# Scale the features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))

print("\nClassification Report - Logistic Regression:\n", classification_report(y_test, log_preds, zero_division=1))
print("\nClassification Report - Decision Tree:\n", classification_report(y_test, tree_preds, zero_division=1))

cm = confusion_matrix(y_test, log_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_model.classes_)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.grid()
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_prob))