from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential



ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription ID>",
    resource_group_name="<resource group name>",
    workspace_name="<workspace name>"
)

PIPELINE_NAME = "fraud-detection-pipeline"


#find the latest completed pipeline job
def get_latest_completed_pipeline_job(ml_client, pipeline_name):
    jobs = ml_client.jobs.list()

    pipeline_jobs = [
        job for job in jobs
        if job.display_name == pipeline_name and job.status == "Completed"
    ]

    if not pipeline_jobs:
        raise RuntimeError(
            f"No completed pipeline jobs found for pipeline '{pipeline_name}'"
        )

    latest_job = sorted(
        pipeline_jobs,
        key=lambda j: j.creation_context.created_at,
        reverse=True
    )[0]

    return latest_job


latest_pipeline_job = get_latest_completed_pipeline_job(
    ml_client, PIPELINE_NAME
)

print(f"Using pipeline job: {latest_pipeline_job.name}")


model_output_uri = (
    f"azureml://jobs/{latest_pipeline_job.name}/outputs/model"
)


#register model

model = Model(
    path=model_output_uri,   
    name="fraud-detection-model",
    description="Fraud detection model trained using Azure ML pipeline",
    type="custom_model",
    tags={
        "pipeline": PIPELINE_NAME,
        "job_name": latest_pipeline_job.name,
        "problem": "fraud-detection",
        "framework": "scikit-learn"
    }
)

registered_model = ml_client.models.create_or_update(model)

print("\nModel registered successfully:")
print(f"Name: {registered_model.name}")
print(f"Version: {registered_model.version}")
