import pandas as pd
import yaml
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate(config):
    df = pd.read_csv("data/processed/test.csv")
    target_col = config["target"]

    X_test, y_test = df.drop(columns=[target_col]), df[target_col]
    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    with open("experiments/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred,average="macro"),
    }
    with open("metric.json", "w") as f:
        json.dump(metrics, f)
        

    print(f"✅ Evaluation complete: {results}")

if __name__ == "__main__":
    config = load_config()
    evaluate(config)
