import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    label_mapping = {"yes": 1, "no": 0, "maybe": 2}
    for pmid, item in data.items():
        question = item["QUESTION"]
        contexts = " ".join(item["CONTEXTS"])

        rows.append({
            "pmid": pmid,
            "text": f"{question} {contexts}",
            "label": label_mapping[item["final_decision"]]
        })
    return pd.DataFrame(rows)

# Load data using the predefined function
train_df = load_and_preprocess('data/pqal_fold0/train_set.json')
dev_df = load_and_preprocess('data/pqal_fold0/dev_set.json')

# Text Representation -- TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text'])
X_dev = vectorizer.transform(dev_df['text'])

y_train = train_df['label']
y_dev = dev_df['label']

# Prediction Model -- Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_dev)
print("Accuracy:", accuracy_score(y_dev, predictions))
print(classification_report(y_dev, predictions, target_names=['no', 'yes', 'maybe']))