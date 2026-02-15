from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
import os

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription ID>",
    resource_group_name="<resource group name>",
    workspace_name=",<workspace name>"
)

MODEL_NAME = "fraud-detection-model"
ENDPOINT_NAME = "fraud-endpoint-1"


models = list(ml_client.models.list(name=MODEL_NAME))

if not models:
    raise RuntimeError(f"No models found with name '{MODEL_NAME}'")

latest_model = sorted(models, key=lambda m: int(m.version))[-1]
print(f"Using model: {MODEL_NAME}, version: {latest_model.version}")

# create endpoint
print("Creating/updating endpoint")

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Real-time fraud detection endpoint",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
print(f"Endpoint '{ENDPOINT_NAME}' created/updated.")

#deployment
print("Creating deployment")


code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=f"azureml:{MODEL_NAME}:{latest_model.version}",
    code_configuration=CodeConfiguration(
        code=code_path,
        scoring_script="score.py"
    ),
    environment="<environment name>:1",
    instance_type="<instance type>",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).wait()
print("Deployment 'blue' created.")


#route traffic
print("Routing 100% traffic to blue deployment")

# Get the updated endpoint and set traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

print(f"\nDeployment successful!")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Deployment: blue")
print(f"Model: {MODEL_NAME} v{latest_model.version}")
 
#endpoint details
endpoint_details = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
print(f"\nEndpoint URI: {endpoint_details.scoring_uri}")
print(f"\nTo get the endpoint key, run:")
print(f"az ml online-endpoint get-credentials -n {ENDPOINT_NAME} -g fraud-detection -w fraud-detection-ws")



