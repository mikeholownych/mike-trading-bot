import argparse
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_data(input_path):
    """
    Load data from S3 or local directory.
    Supports JSONL or Parquet format.
    """
    if input_path.endswith(".jsonl"):
        print(f"Loading JSONL data from {input_path}")
        return pd.read_json(input_path, lines=True)
    elif input_path.endswith(".parquet"):
        print(f"Loading Parquet data from {input_path}")
        return pd.read_parquet(input_path)
    else:
        raise ValueError("Unsupported file format. Use JSONL or Parquet.")

def preprocess_data(df):
    """
    Preprocess the data by encoding categorical variables and splitting features/labels.
    """
    # Encoding target labels
    target_map = {"LONG": 1, "SHORT": -1, "HOLD": 0}
    df["signal"] = df["signal"].map(target_map)
    
    # Dropping irrelevant columns
    X = df.drop(columns=["timestamp", "signal"])
    y = df["signal"]
    
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest model.
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, output_path):
    """
    Save the trained model to the output directory.
    """
    print(f"Saving model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.joblib")
    joblib.dump(model, model_path)

def main():
    """
    Main training script.
    """
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the model")
    args = parser.parse_args()
    
    # Load and preprocess data
    df = load_data(args.train)
    X, y = preprocess_data(df)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model on validation data...")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    # Save the model
    save_model(model, args.model_dir)

if __name__ == "__main__":
    main()
