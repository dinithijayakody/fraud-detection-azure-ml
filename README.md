# Fraud Detection System using Azure Machine Learning

## Simple Overview

This project builds an end-to-end fraud detection system using Azure Machine Learning. It detects fraudulent credit card transactions from highly imbalanced transaction data and deploys the trained model as a real-time REST API in the cloud.

---

# Description

This project demonstrates the complete machine learning lifecycle for a real-world financial fraud detection problem. The system begins with exploratory data analysis to understand the dataset and identify challenges such as severe class imbalance. Appropriate preprocessing strategies, including stratified data splitting and feature scaling, are applied to improve minority class detection.

A Logistic Regression model is trained using class weighting to handle imbalance and evaluated using metrics suitable for imbalanced classification problems such as Recall and ROC-AUC. The workflow is fully automated using Azure ML Pipelines, ensuring modularity, reproducibility, and experiment tracking.

The trained model is registered in Azure ML Model Registry and deployed as a managed online endpoint, allowing real-time inference through a REST API. This project highlights cloud-based machine learning engineering and production deployment practices.

---

# Getting Started

## Dependencies

Before running this project, ensure you have:

- Python 3.9 or higher  
- Windows 10 / macOS / Linux  
- Azure subscription  
- Azure ML Workspace  
- Azure CLI installed  
- pip (Python package manager)

### Required Python Libraries

Installed via `requirements.txt`:

- azure-ai-ml  
- azure-identity  
- scikit-learn  
- pandas  
- numpy  
- joblib  

---

# Installing

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd fraud-detection-azure-ml
```


### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```



### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Azure Login
```bash
az login



```
### 5. Update Pipeline Configuration
open: pipelin/pipeline.py, src/register_model.py, deployment/deploy.py 
and replace placeholders:

- <subscription ID>

- <resource group name>

- <workspace name>

- <compute target name>

- <environment name>:1

- <path to raw dataset in blob storage>


### 6. Executing Program
Step 1: Sibmit Azure ML Pipeline

```bash
python pipelin/pipeline.py
```

Step 2: Register the Model
```bash
python pipeline/register_model.py
```
Step 3: Deploy the model as Online Endpoint
```bash
python deployment/deploy.py
```

Step 4. Test the Endpoint

open: src/test_endpoint.py
and replace placeholders:
- <end point uri>
- <api key>
```bash
python src/test_endpoint.py
```


##Project Structure

```
fraud-detection-azure-ml/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── score.py
    ├── test_endpoint.py
│
├── pipeline/
│   ├── pipeline.py
│   └── register_model.py
│
├── deployment/
│   └── deploy.py
│── environment/
    ├── environment.yml
├── notebooks/
    ├── eda.ipynb

├── requirements.txt
└── README.md
```


Note: Ensure that the dataset is registered as a Data Asset and the environment is registered in Azure ML before running the pipeline.

## What This Project Demonstrates
```
Handling imbalanced classification problems

Azure ML pipeline automation

Model versioning and registry

Managed online endpoint deployment

Real-time REST API inference
```