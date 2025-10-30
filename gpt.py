import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details below to predict loan approval.")


# NAVIGATION

page = st.sidebar.radio("Navigate", ["🔮 Prediction", "📊 Dashboard"])


# PAGE 1: PREDICTION

if page == "🔮 Prediction":

    # --- Loading model pipeline ---
    model_path = "loan_prediction_pipeline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            # Loading the dictionary containing model and features
            export_object = pickle.load(file)
            model_pipeline = export_object['model']
            feature_names = export_object['features'] #  Storing feature names
    else:
        st.error("Model file not found. Please ensure 'loan_prediction_pipeline.pkl' exists.")
        st.stop()

    # --- Input fields ---
    col1, col2 = st.columns(2)

    with col1:
        # Setting a more realistic minimum income
        income = st.number_input(
            "Applicant Income", 
            min_value=500.0, 
            value=5000.0, 
            format="%.2f",
            help="Monthly or Annual Income of the applicant (depends on your model training data)." 
        )
        credit_score = st.number_input(
            "Credit Score", 
            min_value=300.0, 
            max_value=900.0, 
            value=650.0, 
            format="%.2f",
            help="FICO-like score. Higher is better." 
        )
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=300.0, format="%.2f")
        self_employed = st.selectbox("Self-Employed?", ["Yes", "No"])

    # --- Predict button ---
    if st.button("🔍 Predict Loan Approval"):
        raw_data = {
            'ApplicantIncome': income,
            'LoanAmount': loan_amount,
            'CreditScore': credit_score,
            'Education': education,
            'SelfEmployed': self_employed
        }

        input_df = pd.DataFrame([raw_data])

        # --- Prediction ---
        result = model_pipeline.predict(input_df)
        probability = model_pipeline.predict_proba(input_df)[0][1]

        # --- Display prediction ---
        if result[0] == 1:
            st.success(f" Loan is likely to be **APPROVED** ({probability*100:.2f}% confidence)")
        else:
            st.error(f" Loan is likely to be **REJECTED** ({(1-probability)*100:.2f}% confidence)")

        # --- Displaying Feature Importance  ---
        st.subheader("💡 What Influenced the Decision?")
        try:
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']

            all_feature_names = preprocessor.get_feature_names_out()
            cleaned_feature_names = [name.split('__')[-1] for name in all_feature_names]
            
            if isinstance(classifier, (RandomForestClassifier, DecisionTreeClassifier)):
                importances = classifier.feature_importances_
                title = f"Feature Importance (using {classifier.__class__.__name__})"
                
            elif isinstance(classifier, LogisticRegression):
                importances = abs(classifier.coef_[0])
                title = "Feature Coefficients (using Logistic Regression)"
                
            else:
                st.info("Feature influence visualization is not available for this model type.")
                raise Exception("Unsupported classifier for importance plot.") 
                
            if len(importances) != len(cleaned_feature_names):
                st.error(f"Internal Error: Feature count mismatch! Model has {len(importances)} features, but preprocessing found {len(cleaned_feature_names)}.")
                raise Exception("Feature count mismatch error.")
            
            feature_df = pd.DataFrame({
                'Feature': cleaned_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(8, len(feature_df) * 0.4 + 1))
            ax.barh(feature_df['Feature'], feature_df['Importance'], color='teal')
            ax.set_title(title, fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            st.pyplot(fig)
            
        except Exception as e:
            if 'unsupported' not in str(e).lower() and 'mismatch' not in str(e).lower():
                st.error(f"An error occurred while analyzing feature influence: {e}")


        # --- Saving input + prediction to CSV ---
        save_path = "user_inputs.csv"
        input_df["Prediction"] = result
        input_df["Confidence"] = probability

        if os.path.exists(save_path):
            input_df.to_csv(save_path, mode="a", header=False, index=False)
        else:
            input_df.to_csv(save_path, index=False)



# PAGE 2: DASHBOARD

elif page == "📊 Dashboard":
    st.header("📊 Loan Prediction Dashboard")

    if os.path.exists("user_inputs.csv"):
        df = pd.read_csv("user_inputs.csv")

        st.subheader("📋 All User Inputs")
        st.dataframe(df, use_container_width=True)

        # --- Filter options ---
        with st.expander("🔎 Filter Results"):
            filter_choice = st.selectbox("Filter by Prediction", ["All", "Approved (1)", "Rejected (0)"])
            if filter_choice == "Approved (1)":
                df = df[df["Prediction"] == 1]
            elif filter_choice == "Rejected (0)":
                df = df[df["Prediction"] == 0]
        
       
        
        # --- Charts ---
        st.subheader("📈 Summary Charts")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Approval vs Rejection Count**")
            prediction_counts = df["Prediction"].map({1: "Approved", 0: "Rejected"}).value_counts()
            st.bar_chart(prediction_counts)
        with col2:
            st.write("**Average Applicant Income by Prediction**")
            avg_income = df.groupby("Prediction")["ApplicantIncome"].mean()
            
            fig_income, ax_income = plt.subplots()
            avg_income.plot(kind='bar', ax=ax_income, color=['skyblue', 'lightcoral'])
            ax_income.set_xticklabels(['Rejected (0)', 'Approved (1)'], rotation=0)
            ax_income.set_xlabel("Prediction")
            ax_income.set_ylabel("Average Income")
            ax_income.set_title("Average Applicant Income by Prediction")
            st.pyplot(fig_income)
        
        st.write("**Prediction Confidence Distribution**")
        fig, ax = plt.subplots()
        df["Confidence"].hist(bins=10, ax=ax)
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # --- Download button ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Filtered Data as CSV",
            data=csv,
            file_name="loan_predictions_data.csv",
            mime="text/csv"
        )

    else:
        st.info("No data found yet! Run some predictions first.")