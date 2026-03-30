import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

NGRAM_RANGE  = (1, 2)
MAX_FEATURES = None
SUBLINEAR_TF = True
MIN_DF       = 1
C            = 10
CLASS_WEIGHT = 'balanced'
REPEAT_Q     = 2
CTX_LIMIT    = 3

def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    label_mapping = {"yes": 1, "no": 0, "maybe": 2}
    for pmid, item in data.items():
        question = item["QUESTION"]
        contexts = item["CONTEXTS"] if CTX_LIMIT is None else item["CONTEXTS"][:CTX_LIMIT]
        rows.append({
            "pmid": pmid,
            "text": f"{question} " * REPEAT_Q + " ".join(contexts),
            "label": label_mapping[item["final_decision"]]
        })
    return pd.DataFrame(rows)

all_accuracies = []
all_reports    = []

for fold in range(10):
    fold_dir = f'data/pqal_fold{fold}'
    train_df = load_and_preprocess(f'{fold_dir}/train_set.json')
    dev_df   = load_and_preprocess(f'{fold_dir}/dev_set.json')

    vectorizer = TfidfVectorizer(
        stop_words  = 'english',
        ngram_range = NGRAM_RANGE,
        max_features= MAX_FEATURES,
        sublinear_tf= SUBLINEAR_TF,
        min_df      = MIN_DF,
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    X_dev   = vectorizer.transform(dev_df['text'])

    model = LogisticRegression(
        max_iter     = 1000,
        C            = C,
        class_weight = CLASS_WEIGHT,
    )
    model.fit(X_train, train_df['label'])

    preds = model.predict(X_dev)
    acc   = accuracy_score(dev_df['label'], preds)
    all_accuracies.append(acc)
    all_reports.append(classification_report(
        dev_df['label'], preds,
        target_names=['no', 'yes', 'maybe'],
        output_dict=True,
        zero_division=0
    ))
    print(f"Fold {fold:2d} | Accuracy: {acc:.4f}")

print("\n10-Fold Summary:")
print(f"Mean Accuracy : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
for label in ['no', 'yes', 'maybe']:
    f1s = [r[label]['f1-score'] for r in all_reports]
    print(f"Mean F1 [{label:>5s}]: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
macro_f1s = [r['macro avg']['f1-score'] for r in all_reports]
print(f"Mean Macro-F1 : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")