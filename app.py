import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ✅ Load the dataset
file_path = "behavioral_economics_dataset.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# ✅ Map grades to numeric labels
grade_mapping = {grade: i for i, grade in enumerate(df["Faculty_Grade"].unique())}
df["Label"] = df["Faculty_Grade"].map(grade_mapping)

# ✅ Tokenizer and Model
MODEL_NAME = "bert-base-uncased"  # Using a classification model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(grade_mapping))

# ✅ Custom Dataset
class GradingDataset(Dataset):
    def __init__(self, df):
        self.texts = list(df["Concept"] + ". " + df["Student_Response"])
        self.labels = list(df["Label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ✅ Prepare dataset
dataset = GradingDataset(df)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# ✅ Train the model
trainer.train()

# ✅ Streamlit UI
st.title("Automated Student Response Grading")
st.write("Enter the concept and student's response to get an estimated grade.")

concept = st.text_input("Concept")
student_answer = st.text_area("Student's Answer", height=200)

if st.button("Predict Grade"):
    if concept and student_answer:
        input_text = f"{concept}. {student_answer}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits).item()
        predicted_grade = {v: k for k, v in grade_mapping.items()}[pred_label]
        st.success(f"Predicted Grade: {predicted_grade}")
    else:
        st.warning("Please enter both the concept and student response.")
