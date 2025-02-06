import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize_data(inputs, tokenizer, max_length=64):
    input_encodings = tokenizer(
        list(inputs), max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    return input_encodings

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

def generate_response(input_text):
    # Prepare input
    test_inputs = [input_text]
    test_inputs_enc = tokenize_data(test_inputs, tokenizer)
    test_dataset = CustomDataset(test_inputs_enc)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                repetition_penalty=1.2,  
                max_length=128
            )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return prediction

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)