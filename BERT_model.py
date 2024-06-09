import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from html import unescape
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.pipeline import Pipeline


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_text(text_data):
    encoded = tokenizer.batch_encode_plus(
        text_data,
        max_length=512,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    )
    return encoded

balance_sheet_docs = []
cash_flow_docs = []
income_statement_docs = []
notes_docs = []
other_docs = []

for folder_name in ['Balance Sheets', 'Cash Flow', 'Income Statement', 'Notes', 'Others']:
    folder_path = os.path.join(rf"F:\my_own_projects\Finac_guvi\data", folder_name)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            match = re.search(r'<document_content>(.*?)</document_content>', text, re.DOTALL)
            if match:
                text_content = match.group(1)
                cleaned_text = re.sub(r'<[^>]+>', '', text_content)
                cleaned_text = unescape(cleaned_text)
            else:
                cleaned_text = text
            if folder_name == 'Balance Sheets':
                balance_sheet_docs.append(cleaned_text)
            elif folder_name == 'Cash Flow':
                cash_flow_docs.append(cleaned_text)
            elif folder_name == 'Income Statement':
                income_statement_docs.append(cleaned_text)
            elif folder_name == 'Notes':
                notes_docs.append(cleaned_text)
            else:
                other_docs.append(cleaned_text)

# Step 2: Text Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return ' '.join(tokens)

balance_sheet_docs = [preprocess_text(doc) for doc in balance_sheet_docs]
cash_flow_docs = [preprocess_text(doc) for doc in cash_flow_docs]
income_statement_docs = [preprocess_text(doc) for doc in income_statement_docs]
notes_docs = [preprocess_text(doc) for doc in notes_docs]
other_docs = [preprocess_text(doc) for doc in other_docs]

# Step 3: Combining Documents and Labels
documents = []
labels = []


for doc in balance_sheet_docs:
    documents.append(doc)
    labels.append(0)

for doc in cash_flow_docs:
    documents.append(doc)
    labels.append(1)

for doc in income_statement_docs:
    documents.append(doc)
    labels.append(2)

for doc in notes_docs:
    documents.append(doc)
    labels.append(3)

for doc in other_docs:
    documents.append(doc)
    labels.append(4)


#Create dataset for model training and testing:
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)
train_encoded = encode_text(X_train)
test_encoded = encode_text(X_test)

train_dataset = TensorDataset(train_encoded['input_ids'], train_encoded['attention_mask'], torch.tensor(y_train))
test_dataset = TensorDataset(test_encoded['input_ids'], test_encoded['attention_mask'], torch.tensor(y_test))

batch_size = 32
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluation loop
model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in test_loader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    tmp_eval_loss, logits = outputs.loss, outputs.logits
    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    nb_eval_examples += input_ids.size(0)
    if nb_eval_steps % 100 == 0:
        print(f'Step: {nb_eval_steps}, Loss: {eval_loss/nb_eval_steps}')
    print(f'Evaluation Loss: {eval_loss/nb_eval_steps}')