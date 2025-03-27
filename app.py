
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Define the correct model name
MODEL_NAME = "bert-base-uncased"  # Use a classification-friendly model

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)

# ✅ Grade mapping (Ensure labels match dataset)
grade_mapping = {0: "A+", 1: "A", 2: "A-", 3: "B+", 4: "B", 5: "B-", 6: "C+", 7: "C", 8: "D", 9: "F"}

# ✅ Function to predict grade
def predict_grade(concept, student_response):
    combined_input = f"Concept: {concept}. Student Answer: {student_response}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return grade_mapping.get(predicted_class, "Unknown")

# ✅ Streamlit UI
st.title("📚 AI-Powered Student Grading")
st.write("Enter the concept and the student's response to predict their grade.")

# ✅ Input fields
concept = st.text_input("🧠 Concept (What was taught?)")
student_answer = st.text_area("📝 Student's Answer", height=200)

# ✅ Button to trigger prediction
if st.button("🎯 Predict Grade"):
    if concept and student_answer:
        predicted_grade = predict_grade(concept, student_answer)
        st.success(f"✅ Predicted Grade: **{predicted_grade}**")
    else:
        st.warning("⚠️ Please enter both the concept and student response.")
