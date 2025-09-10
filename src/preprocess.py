import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess(config):
    df = pd.read_csv(config["dataset"])
    target_col = config["target"]

    # Drop irrelevant Titanic columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Fill missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical
    if config["preprocessing"]["encode"]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Scale numeric features (exclude target)
    if config["preprocessing"]["scale"]:
        scaler = StandardScaler()
        num_cols = df.drop(columns=[target_col]).select_dtypes(include="number").columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Train-test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"]
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Preprocessing done: train.csv, test.csv created")

if __name__ == "__main__":
    config = load_config()
    preprocess(config)
