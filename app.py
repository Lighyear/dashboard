import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- Load models and expected features ---
rf_model = joblib.load("random_forest_model_H.pkl")
cat_model = joblib.load("catboost_model_H.pkl")
X_test = joblib.load("X_test.pkl")  # Make sure this is just the feature data
expected_features = joblib.load("expected_features.pkl")  # The features used during training

# --- Define selected important features (for display) ---
selected_features = [
    'ZUx - Cycle time',
    'ZDx - Plasticizing time',
    'Mold temperature',
    'APVs - Specific injection pressure peak value',
    'SVo - Shot volume',
    'CPn - Screw position at the end of hold pressure',
    'time_to_fill'
]

# --- Label mapping ---
label_mapping = {0: "Waste", 1: "Acceptable", 2: "Inefficient", 3: "Target"}

# --- Streamlit UI ---
st.title(" Quality Prediction Dashboard")
st.markdown("Predict part quality using trained ML models and selected process features.")

# --- Model selection ---
model_choice = st.radio("Choose a Model", ["Random Forest", "CatBoost"])

# --- Predict button ---
if st.button("Run Prediction"):
    X_input = X_test.copy()

    # Align features: Add missing, drop extras, and reorder to expected_features
    for col in expected_features:
        if col not in X_input.columns:
            X_input[col] = 0  # or use default/mean if available

    # Reorder columns to match training
    X_input = X_input[expected_features]

    # Select model and predict
    model = rf_model if model_choice == "Random Forest" else cat_model
    predictions = model.predict(X_input)

    # Prepare result DataFrame
    results_df = X_input.copy()
    results_df["Predicted Quality"] = predictions
    results_df["Predicted Label"] = results_df["Predicted Quality"].map(label_mapping)

    # Show predictions (with selected features)
    st.write("# Sample Predictions (Important Features)")
    st.dataframe(results_df[selected_features + ["Predicted Label"]].head())

    # Plot predicted label distribution
    st.write("# Predicted Quality Distribution")
    counts = results_df["Predicted Label"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Predicted Class Distribution")
    st.pyplot(fig)

    # Download results
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Predictions CSV", csv, "predicted_results.csv", "text/csv")

else:
    st.info("Click the button above to run prediction using the selected model.")
