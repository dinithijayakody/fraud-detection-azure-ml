import argparse
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(input_path, output_path, test_size=0.2,random_state=42):
    """
    Preprocess credit card fraud dataset:
    - Scale 'Amount'
    -Stratified train-test split
    -Save processed datasets

    """
    #Load the dataset
    df = pd.read_csv(input_path)

    #seperate features and target
    X = df.drop("Class",axis=1)
    y = df["Class"]

    #Scale 'Amount' feature
    scaler =StandardScaler()
    X["Amount"]=scaler.fit_transform(X[["Amount"]])

    #train-test split - stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state)
    
    #Create output directory
    os.makedirs(output_path, exist_ok=True)

    #save processed datasets
    X_train.to_csv(os.path.join(output_path,"X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path,"y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path,"y_test.csv"), index=False)

    #save the scaler for inference
    joblib.dump(scaler, os.path.join(output_path,"scaler.joblib"))
    print("Preprocessing completed successfully.")

if __name__ =="__main__":
    parser =argparse.ArgumentParser()

    parser.add_argument("--input_path",type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path",type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    preprocess_data(
        input_path=args.input_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )



