import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

# Train the model
@st.cache_data
def train_model():
    X = df[['reading score', 'writing score']]
    y = df['math score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

st.title("Math Score Predictor")

# User inputs
reading_score = st.number_input("Enter Reading Score", min_value=0, max_value=100, value=70)
writing_score = st.number_input("Enter Writing Score", min_value=0, max_value=100, value=70)

if st.button("Predict Math Score"):
    input_data = pd.DataFrame([[reading_score, writing_score]], columns=['reading score', 'writing score'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Math Score: {prediction:.2f}")

# Optionally display dataset overview for users
#if st.checkbox("Show dataset overview"):
 #   st.write(df.head())

  #  st.write(df.describe())
