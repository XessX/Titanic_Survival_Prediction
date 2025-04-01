import streamlit as st
import pandas as pd
import joblib

# Load trained model, scaler, and expected columns
model = joblib.load("titanic_random_forest_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")
expected_columns = joblib.load("titanic_feature_columns.pkl")

# --- Streamlit UI ---
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict if a passenger would survive the Titanic disaster.")

# --- Inputs ---
pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Age group logic
if age <= 12:
    age_group = "Child"
elif age <= 19:
    age_group = "Teen"
elif age <= 40:
    age_group = "Adult"
elif age <= 60:
    age_group = "Middle-aged"
else:
    age_group = "Senior"

# --- Prepare input ---
input_data = {
    'Pclass': pclass,
    'Sex': 0 if sex == "male" else 1,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked_Q': 1 if embarked == 'Q' else 0,
    'Embarked_S': 1 if embarked == 'S' else 0,
    'AgeGroup_Teen': 1 if age_group == "Teen" else 0,
    'AgeGroup_Adult': 1 if age_group == "Adult" else 0,
    'AgeGroup_Middle-aged': 1 if age_group == "Middle-aged" else 0,
    'AgeGroup_Senior': 1 if age_group == "Senior" else 0
}

# Fill any missing expected columns
for col in expected_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame & reorder
df_input = pd.DataFrame([input_data])[expected_columns]

# Scale the input
df_scaled = scaler.transform(df_input)

# --- Predict ---
if st.button("Predict Survival"):
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    if prediction == 1:
        st.success(f"🎉 Survived! (Confidence: {probability:.2%})")
    else:
        st.error(f"💀 Did not survive. (Confidence: {1 - probability:.2%})")
