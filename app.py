import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

# ✅ Load the correct model (Masked LM)
MODEL_NAME = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# ✅ Use a fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# ✅ Function to check response relevance
def check_relevance(concept, student_response):
    combined_input = f"{concept}. {student_response}"
    
    # Tokenize input
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    
    # Get model predictions (Masking a random token to check contextual relevance)
    mask_token = tokenizer.mask_token
    test_sentence = f"{concept}. {mask_token} {student_response}"  # Insert [MASK] for relevance check
    predictions = fill_mask(test_sentence)

    # Extract top predicted words
    top_words = [pred["token_str"] for pred in predictions]

    # Basic relevance score based on prediction confidence
    score = sum(pred["score"] for pred in predictions) / len(predictions)

    return f"Relevance Score: {round(score * 100, 2)}%", top_words

# ✅ Streamlit UI
st.title("Student Answer Relevance Check")
st.write("Enter the concept and student's response to check relevance:")

concept = st.text_input("Concept (What was taught?)")
student_answer = st.text_area("Student's Answer", height=200)

if st.button("Check Relevance"):
    if concept and student_answer:
        relevance_score, suggested_words = check_relevance(concept, student_answer)
        st.success(f"{relevance_score}")
        st.write(f"Suggested Words for Improvement: {', '.join(suggested_words)}")
    else:
        st.warning("Please enter both the concept and student response.")
