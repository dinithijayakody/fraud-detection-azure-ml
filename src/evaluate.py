import argparse
import os
import json
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    roc_auc_score,  
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)

def evaluate_model(data_path, model_path, output_path):
    """Evaluate the trained fraud detection model."""

    #Load test data
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()

    #Load the trained model
    model = joblib.load(model_path)

    #Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    }

    print("\nModel Evaluation Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics
    os.makedirs(output_path, exist_ok=True)

    metrics_path = os.path.join(output_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str,required=True,help="Path to processed data directory"
    )

    parser.add_argument("--model_path",type=str,required=True,help="Path to trained model file"
    )

    parser.add_argument("--output_path",type=str,required=True,help="Directory to save evaluation outputs"
    )

    args = parser.parse_args()

    evaluate_model(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path
    )
