import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login

# 🔹 Authenticate with Hugging Face (Streamlit Secrets)
try:
    HF_TOKEN = st.secrets["huggingface"]["token"]
    login(token=HF_TOKEN)
    st.success("✅ Successfully authenticated with Hugging Face!")
except Exception as e:
    st.error(f"⚠️ Authentication failed: {e}")
    st.stop()

# 🔹 Load fine-tuned model from Hugging Face
MODEL_NAME = "your-huggingface-username/your-grading-model"  # Change this!

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# 🔹 Grade Mapping
grade_mapping = {0: "A+", 1: "A", 2: "B", 3: "C", 4: "D", 5: "F"}

# 🔹 Prediction Function (Uses both concept & student response)
def predict_grade(concept, student_response):
    input_text = f"Concept: {concept} [SEP] Student Answer: {student_response}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return grade_mapping.get(predicted_class, "Unknown")

# 🔹 Streamlit UI
st.title("📚 Student Grade Predictor")
st.write("Enter the expected concept and the student's response to predict the grade.")

# 🔹 Input Fields
concept = st.text_area("✅ Expected Answer (Concept)", height=150)
student_answer = st.text_area("✍️ Student's Answer", height=150)

# 🔹 Predict Button
if st.button("Predict Grade"):
    if concept.strip() and student_answer.strip():
        predicted_grade = predict_grade(concept, student_answer)
        st.success(f"🎯 Predicted Grade: **{predicted_grade}**")
    else:
        st.warning("⚠️ Please enter both the expected concept and student response.")
