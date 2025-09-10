import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train(config):
    df = pd.read_csv("data/processed/train.csv")
    target_col = config["target"]
    X, y = df.drop(columns=[target_col]), df[target_col]

    # Select model
    if config["model"]["type"] == "RandomForestClassifier":
        model = RandomForestClassifier(**config["model"]["params"])
    elif config["model"]["type"] == "LogisticRegression":
        model = LogisticRegression(**config["model"]["params"])
    else:
        raise ValueError("Unsupported model")

    # Train
    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model trained & saved at models/model.pkl")

if __name__ == "__main__":
    config = load_config()
    train(config)
