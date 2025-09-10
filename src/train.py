import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import json

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train(config):
    df = pd.read_csv("data/processed/train.csv")
    target_col = config["target"]
    X, y = df.drop(columns=[target_col]), df[target_col]

    df = pd.read_csv("data/processed/test.csv")
    target_col = config["target"]
    X_test, y_test = df.drop(columns=[target_col]), df[target_col]

    # Select model
    if config["model"]["type"] == "RandomForestClassifier":
        model = RandomForestClassifier(**config["model"]["params"])
    elif config["model"]["type"] == "LogisticRegression":
        
# Assuming config["model"]["params"]["max_iter"] is defined elsewhere
        max_iter = config["model"]["params"]["max_iter"]

        # Create a logistic regression model
        lr = LogisticRegression(max_iter=max_iter, warm_start=True, random_state=42,l1_ratio=0.7,solver='saga',penalty='elasticnet')

        # Dictionary to hold all metrics per iteration
        metrics = []

        for i in range(1, max_iter + 1):
            lr.max_iter = i
            lr.fit(X, y)

            # Make predictions
            y_train_pred = lr.predict(X)
            y_train_pred_proba = lr.predict_proba(X)
            y_test_pred = lr.predict(X_test)
            y_test_pred_proba = lr.predict_proba(X_test)

            # Calculate metrics
            train_accuracy = accuracy_score(y, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_loss = log_loss(y, y_train_pred_proba)
            test_loss = log_loss(y_test, y_test_pred_proba)

            # Store metrics for this iteration
            metrics.append({
            "iteration": i,
            "train_accuracy": accuracy_score(y, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "train_loss": log_loss(y, y_train_pred_proba),
            "test_loss": log_loss(y_test, y_test_pred_proba)
        })

        # Save metrics to log.json
        with open("log.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("Metrics saved to log.json")
        model = lr
    
    else:
        raise ValueError("Unsupported model")

    # Train
    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model trained & saved at models/model.pkl")

if __name__ == "__main__":
    config = load_config()
    train(config)
