import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json


def load_data(file_path):
    """
    Load training data from the specified path.
    Supports JSON Lines format.
    """
    print(f"Loading data from {file_path}")
    if file_path.endswith(".jsonl"):
        data = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use JSONL.")
    return data


def preprocess_data(data):
    """
    Preprocess the data for training.
    """
    print("Preprocessing data...")

    # Map signal to numeric values
    target_map = {"LONG": 1, "SHORT": -1, "HOLD": 0}
    if "signal" not in data.columns:
        raise ValueError("The 'signal' column is missing from the dataset.")
    
    data["signal"] = data["signal"].map(target_map)

    # Drop irrelevant columns
    X = data.drop(columns=["timestamp", "signal"])
    y = data["target"]

    print(f"Data preprocessing complete. Features: {X.columns.tolist()}")
    return X, y


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    """
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model on validation data.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Model evaluation complete.")
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))


def save_model(model, model_dir):
    """
    Save the trained model to the specified directory.
    """
    print(f"Saving model to {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")


def main():
    """
    Main function for training the model.
    """
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to the training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()

    # Load data
    train_data_path = os.path.join("/opt/ml/input/data/train", args.train)
    data = load_data(train_data_path)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Save the trained model
    save_model(model, args.model_dir)


if __name__ == "__main__":
    main()
