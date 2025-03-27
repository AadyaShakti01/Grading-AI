import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login

# 🔹 Authenticate with Hugging Face (Read from Streamlit Secrets)
try:
    HF_TOKEN = st.secrets["huggingface"]["token"]
    login(token=HF_TOKEN)
    st.success("✅ Successfully authenticated with Hugging Face!")
except Exception as e:
    st.error(f"⚠️ Authentication failed: {e}")
    st.stop()

# 🔹 Load your fine-tuned model from Hugging Face
MODEL_NAME = "your-huggingface-username/your-grading-model"  # Replace with actual model

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# 🔹 Grade Mapping
grade_mapping = {0: "A+", 1: "A", 2: "B", 3: "C", 4: "D", 5: "F"}

# 🔹 Prediction Function
def predict_grade(student_response):
    inputs = tokenizer(student_response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return grade_mapping.get(predicted_class, "Unknown")

# 🔹 Streamlit UI
st.title("📚 Student Grade Predictor")
st.write("Enter a student's response below, and the model will predict the grade.")

# 🔹 Input Box
student_answer = st.text_area("Student's Answer", height=200)

# 🔹 Predict Button
if st.button("Predict Grade"):
    if student_answer.strip():
        predicted_grade = predict_grade(student_answer)
        st.success(f"🎯 Predicted Grade: **{predicted_grade}**")
    else:
        st.warning("⚠️ Please enter a student response.")
