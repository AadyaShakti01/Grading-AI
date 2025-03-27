import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Load dataset & extract unique concepts
DATASET_PATH = "/mnt/data/behavioral_economics_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# ✅ Ensure 'Concept' column exists
if "Concept" in df.columns:
    unique_concepts = df["Concept"].dropna().unique().tolist()
else:
    st.error("🚨 'Concept' column not found in dataset! Please check the file.")
    st.stop()

# ✅ Load model & tokenizer
MODEL_NAME = "bert-base-uncased"  # Ensure correct model is used
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)

# ✅ Grade mapping (Ensure it aligns with model output)
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
st.write("Select the concept from the dropdown and enter the student's response to predict their grade.")

# ✅ Dropdown for concept selection (NOW VISIBLE)
concept = st.selectbox("Select Concept" options=df["Concept"].unique())

# ✅ Text area for student's answer
student_answer = st.text_area("📝 Student's Answer", height=150)

# ✅ Predict button
if st.button("🎯 Predict Grade"):
    if student_answer:
        predicted_grade = predict_grade(selected_concept, student_answer)
        st.success(f"✅ Predicted Grade: **{predicted_grade}**")
    else:
        st.warning("⚠️ Please enter the student's response.")
