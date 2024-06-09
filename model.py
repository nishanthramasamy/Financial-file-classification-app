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


#Lists to store the extracted texts from the documents
balance_sheet_docs = []
cash_flow_docs = []
income_statement_docs = []
notes_docs = []
other_docs = []

#Extracting data from all files
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

#Text Preprocessing
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

#Combining Documents and Labels
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


#Additional step: Create a dataframe for more clear understanding:
# data = {
#         'document' : documents,
#         'label' : labels
# }

# df = pd.DataFrame(data, columns=['document', 'label'])

#Train and test the model:
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
#logistic_regression_model = LogisticRegression()
Random_classifier_model = RandomForestClassifier(n_estimators=100, max_depth=100)
pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', Random_classifier_model)])
pipeline.fit(X_train, y_train)


#serialize model for easy imgration
with open('random_forest.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('random_forest.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

#Predict with text data:
y_pred = loaded_data.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")