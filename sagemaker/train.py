import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json


def load_data(file_path):
    """
    Load training data from a JSONL file.
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith(".jsonl"):
        data = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Expected JSONL.")
    
    print(f"Data loaded successfully. Shape: {data.shape}")
    return data


def preprocess_data(data):
    """
    Preprocess the data for training.
    Maps 'signal' to numeric 'target' and ensures 'target' is a list for compatibility.
    """
    print("Preprocessing data...")

    # Map 'signal' to numeric values if 'target' is not already present
    target_map = {"LONG": 1, "SHORT": -1, "HOLD": 0}
    if "signal" in data.columns and "target" not in data.columns:
        data["target"] = data["signal"].map(target_map)

    if "target" not in data.columns:
        raise ValueError("The 'target' field is missing from the dataset.")

    # Ensure 'target' is a list for compatibility
    data["target"] = data["target"].apply(lambda x: [x] if isinstance(x, int) else x)

    # Drop irrelevant columns
    X = data.drop(columns=["timestamp", "signal", "target"], errors="ignore")
    y = data["target"]

    print(f"Preprocessing complete. Features: {X.columns.tolist()}, Target shape: {len(y)}")
    return X, y


def train_model(X_train, y_train):
    """
    Train a Random Forest model on the training data.
    """
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, [t[0] for t in y_train])  # Extract numeric values from the list
    print("Model training complete.")
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model and print metrics.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_val)
    y_val_numeric = [t[0] for t in y_val]  # Extract numeric values from the list
    accuracy = accuracy_score(y_val_numeric, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_val_numeric, y_pred))


def save_model(model, model_dir):
    """
    Save the trained model to the specified directory.
    """
    print(f"Saving model to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")


def main():
    """
    Main entry point for the training script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to the training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()

    # Define the training data path
    train_data_path = os.path.join("/opt/ml/input/data/train", args.train)
    
    # Load and preprocess data
    data = load_data(train_data_path)
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
