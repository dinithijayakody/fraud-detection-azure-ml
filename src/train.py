import argparse
import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def train_model(data_path, model_output_path, random_state=42):
    """
    Train a fraud detection model using Logistic Regression with class imbalance handling.
    """

    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()

    # Define model with class weighting
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict probabilities and labels
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Model Evaluation Metrics:")
    print(f"Precision:{precision}")
    print(f"Recall:{recall}")
    print(f"F1-score:{f1}")
    print(f"ROC-AUC:{roc_auc}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(model_output_path, exist_ok=True)
    model_path = os.path.join(model_output_path, "fraud_model.joblib")
    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",type=str,required=True,help="Path to processed data directory")
    parser.add_argument(
        "--model_output_path",type=str,required=True,help="Directory to save trained model")
    parser.add_argument("--random_state",type=int,default=42)

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output_path,
        random_state=args.random_state)
