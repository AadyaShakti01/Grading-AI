import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("your-trained-model")

def predict_grade(response, concept):
    inputs = tokenizer(f"{concept}: {response}", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    st.write("Logits:", logits.tolist())  # Print raw model output
    st.write("Probabilities:", probs.tolist())  # Print confidence scores
    
    predicted_grade_idx = torch.argmax(probs, dim=1).item()
    grade_mapping = {0: "A+", 1: "A", 2: "A-", 3: "B+", 4: "B", 5: "B-", 6: "C+", 7: "C", 8: "D", 9: "F"}
    
    return grade_mapping[predicted_grade_idx]

st.title("AI-Based Grading System")
response = st.text_input("Enter Student Response:")
concept = st.selectbox("Select Concept", ["Loss Aversion", "Endowment Effect", "Hyperbolic Discounting"])

if st.button("Get Grade"):
    grade = predict_grade(response, concept)
    st.write("Predicted Grade:", grade)
