import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Fraud Detection System – Type I & Type II Error Analysis")

st.write("""
This app trains a Fraud Detection ML model and shows:
- Confusion Matrix  
- Type I Error (False Positives)  
- Type II Error (False Negatives)  
""")

uploaded_file = st.file_uploader("Upload Credit Card Fraud Dataset (creditcard.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Splitting data
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    st.subheader("Confusion Matrix")
    st.write(cm)

    # Error summary
    st.subheader("Error Analysis")
    st.write(f"**Type I Error (False Positive): {fp}**")
    st.write(f"**Type II Error (False Negative): {fn}**")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Heatmap Plot
    st.subheader("Confusion Matrix Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

else:
    st.info("Upload the dataset to start analysis.")
