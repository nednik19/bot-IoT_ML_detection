import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from tabulate import tabulate

# Fancy banner
def print_banner():
    print("=" * 60)
    print("🚀 IoT Botnet Anomaly Detection - Model Evaluation 🚀")
    print("=" * 60)

# Preprocess data function
def preprocess_input_data(data, scaler_path='./saved_model/scaler.pkl'):
    print("\n[INFO] Preprocessing data...")
    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Expected columns based on training
    required_columns = [
        "flgs", "proto", "pkts", "bytes", "dur", "mean", "stddev",
        "sum", "min", "max", "rate"  # Replace with actual columns used in training
    ]

    # Retain only required columns
    data = data[required_columns]

    # Handle missing or mixed types
    data = data.fillna(0)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale the data
    data = scaler.transform(data)
    return data

# Function for incremental evaluation
def evaluate_model_incremental(directory, model_path='./saved_model/iforest_model.pkl', scaler_path='./saved_model/scaler.pkl'):
    # Print banner
    print_banner()

    # Load the model
    model = joblib.load(model_path)

    # Initialize counters
    total_normal = 0
    total_anomalies = 0
    true_normal_pred_normal = 0
    true_attack_pred_attack = 0
    false_normal_pred_attack = 0
    false_attack_pred_normal = 0

    # List files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"\n📂 Found {len(all_files)} CSV files in the directory.")

    for idx, file_path in enumerate(tqdm(all_files, desc="Processing Files"), start=1):
        print(f"\n[INFO] Processing file {idx}/{len(all_files)}: {file_path}")

        # Load the file
        data = pd.read_csv(file_path, low_memory=False)

        # Ensure 'attack' column exists
        if 'attack' not in data.columns:
            print(f"[WARNING] Skipping file {file_path}: 'attack' column missing.")
            continue

        # Extract actual labels
        actual_labels = data['attack'].values
        total_normal += np.sum(actual_labels == 0)
        total_anomalies += np.sum(actual_labels == 1)

        # Preprocess the data
        preprocessed_data = preprocess_input_data(data, scaler_path)

        # Make predictions
        predictions = model.predict(preprocessed_data)
        predictions = np.array([1 if p == -1 else 0 for p in predictions])  # Convert -1 to anomaly (1)

        # Update counters
        true_normal_pred_normal += np.sum((actual_labels == 0) & (predictions == 0))
        true_attack_pred_attack += np.sum((actual_labels == 1) & (predictions == 1))
        false_normal_pred_attack += np.sum((actual_labels == 0) & (predictions == 1))
        false_attack_pred_normal += np.sum((actual_labels == 1) & (predictions == 0))

    # Calculate evaluation metrics
    total_actual_labels = total_normal + total_anomalies
    accuracy = (true_normal_pred_normal + true_attack_pred_attack) / total_actual_labels
    precision = true_attack_pred_attack / (true_attack_pred_attack + false_normal_pred_attack)
    recall = true_attack_pred_attack / (true_attack_pred_attack + false_attack_pred_normal)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Display detailed results
    results = [
        ["Metric", "Value"],
        ["Accuracy", f"{accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"],
        ["True Normal Predicted as Normal", true_normal_pred_normal],
        ["True Attack Predicted as Attack", true_attack_pred_attack],
        ["False Normal Predicted as Attack", false_normal_pred_attack],
        ["False Attack Predicted as Normal", false_attack_pred_normal],
    ]
    print("\n📊 Model Evaluation Results:")
    print(tabulate(results, headers="firstrow", tablefmt="pretty"))

    return {
        "total_normal": total_normal,
        "total_anomalies": total_anomalies,
        "true_normal_pred_normal": true_normal_pred_normal,
        "true_attack_pred_attack": true_attack_pred_attack,
        "false_normal_pred_attack": false_normal_pred_attack,
        "false_attack_pred_normal": false_attack_pred_normal,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Example usage
# dataset_directory = '/Users/nednik/Downloads/Dataset/Entire-Dataset/'  # Replace with your dataset directory
dataset_directory = './data/'
model_path = './saved_model/iforest_model.pkl'
scaler_path = './saved_model/scaler.pkl'

# Uncomment to execute
results = evaluate_model_incremental(dataset_directory, model_path=model_path, scaler_path=scaler_path)
print("\nEvaluation Results:", results)
