from sklearn.metrics import accuracy_score, f1_score, classification_report

labels = ["yes", "no", "maybe"]

def compute_metrics(actual_values, predicted_values):
    accuracy = accuracy_score(actual_values, predicted_values)
    macro_f1 = f1_score(actual_values, predicted_values, average="macro")
    classification_reports = classification_report(actual_values, predicted_values,labels= labels, zero_division= 0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": classification_reports
    }

def print_metrics(actual_values, predicted_values):
    metrics = compute_metrics(actual_values, predicted_values)
    print("Accuracy", metrics['accuracy'])
    print("Macro F1", metrics['macro_f1'])
    print("Classification_report\n")
    print(metrics['classification_report'])
