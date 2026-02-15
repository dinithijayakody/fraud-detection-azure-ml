import requests
import json

# Endpoint details
endpoint_uri = "<endpoint URI from deployment output>"
api_key = "<endpoint API key from deployment output>"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Sample transaction data
data = [
    {
        "Time": 406,
        "V1": -2.312226542,
        "V2": 1.951992011,
        "V3": -1.609850732,
        "V4": 3.997905588,
        "V5": -0.522187865,
        "V6": -1.426545319,
        "V7": -2.537387306,
        "V8": 1.391657248,
        "V9": -2.770089277,
        "V10": -2.772272145,
        "V11": 3.202033207,
        "V12": -2.899907388,
        "V13": -0.595221881,
        "V14": -4.289253782,
        "V15": 0.38972412,
        "V16": -1.14074718,
        "V17": -2.830055675,
        "V18": -0.016822468,
        "V19": 0.416955705,
        "V20": 0.126910559,
        "V21": 0.517232371,
        "V22": -0.035049369,
        "V23": -0.465211076,
        "V24": 0.320198199,
        "V25": 0.044519167,
        "V26": 0.177839798,
        "V27": 0.261145003,
        "V28": -0.143275875,
        "Amount": 0
    }
]

# Make request
print("Sending request to endpoint")
response = requests.post(endpoint_uri, headers=headers, json=data)

print(f"\nStatus Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    result = response.json()
    if "error" in result:
        print(f"\n Error from model: {result['error']}")
    elif "prediction" in result:
        print(f"\n Prediction: {'FRAUD' if result['prediction'][0] == 1 else 'LEGITIMATE'}")
        print(f"Fraud Probability: {result['fraud_probability'][0]:.2%}")
else:
    print(f"\n HTTP Error: {response.text}")





