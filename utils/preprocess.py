import json
import pandas as pd
from sklearn.utils import resample

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

def upsample_no_and_maybe(df):
    df['label'] = df['label'].str.lower().str.strip()
    yes_df = df[df['label'] == "yes"]
    no_df = df[df['label'] == "no"]
    maybe_df = df[df['label'] == "maybe"]

    no_upsampled = resample(
        no_df,
        replace= True,
        n_samples= len(yes_df),
        random_state= 28
    )

    maybe_upsampled = resample(
        maybe_df,
        replace= True,
        n_samples= len(yes_df),
        random_state= 28
    )

    upsampled_df = pd.concat([yes_df, no_upsampled, maybe_upsampled]).sample(frac=1, random_state= 28)
    return upsampled_df