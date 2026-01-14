import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Decode function
# -----------------------------
def decode_emotion(pred):
    emotion_map = {
        0: "Anger 😡",
        1: "Anxiety 😟",
        2: "Boredom 😐",
        3: "Happiness 😄",
        4: "Neutral 🙂",
        5: "Sadness 😢"
    }
    return emotion_map.get(pred, "Unknown")

# -----------------------------
# Load model
# -----------------------------
with open("emotion_xgb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Emotion Prediction App", layout="centered")



# -----------------------------
# App title
# -----------------------------
st.title("📊 Social Media Emotion Prediction")
st.write("Predict **Dominant Emotion** using social media behavior")

st.divider()

# -----------------------------
# Inputs
# -----------------------------
age = st.slider("Age", 10, 100, 25)

gender = st.radio("Gender", ["Male", "Female", "Non-binary"], horizontal=True)

platform = st.selectbox(
    "Platform",
    ["Instagram", "Facebook", "Twitter", "Snapchat", "LinkedIn"]
)

daily_usage = st.number_input("Daily Usage Time (minutes)", 0, 1440, 120)
posts_per_day = st.number_input("Posts Per Day", 0, 100, 2)
likes_per_day = st.number_input("Likes Received Per Day", 0, 5000, 50)
comments_per_day = st.number_input("Comments Received Per Day", 0, 1000, 10)
messages_per_day = st.number_input("Messages Sent Per Day", 0, 5000, 30)

# -----------------------------
# Predict
# -----------------------------
if st.button("🚀 Predict Emotion"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Platform": [platform],
        "Daily_Usage_Time (minutes)": [daily_usage],
        "Posts_Per_Day": [posts_per_day],
        "Likes_Received_Per_Day": [likes_per_day],
        "Comments_Received_Per_Day": [comments_per_day],
        "Messages_Sent_Per_Day": [messages_per_day]
    })

    prediction = model.predict(input_data)
    emotion = decode_emotion(prediction[0])

    st.success(f"🎯 Predicted Dominant Emotion: **{emotion}**")
