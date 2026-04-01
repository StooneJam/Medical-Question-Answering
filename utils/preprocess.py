import json
import pandas as pd


def load_and_preprocess(file_path, label_encoding = True):
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
            "question": f"{question}",
            "context": f"{contexts}",
            "label": label_mapping[item["final_decision"]] if(label_encoding) else item["final_decision"]
        })
    return pd.DataFrame(rows)

