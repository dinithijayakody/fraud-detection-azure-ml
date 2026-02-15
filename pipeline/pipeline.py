from azure.ai.ml import dsl, command, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription ID>",
    resource_group_name="<resource group name>",
    workspace_name="<workspace name>"
)

#pipeline
@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection pipeline (preprocess, train and evaluate)",
    default_compute="<compute target name>"
)
def fraud_pipeline(raw_data):

    #preprocess step
    preprocess_step = command(
        name="preprocess",
        code=".",
        command=(
            "python src/preprocess.py "
            "--input_path ${{inputs.raw_data}} "
            "--output_path ${{outputs.processed_data}}"
        ),
        inputs={
            "raw_data": Input(type="uri_file")
        },
        outputs={
            "processed_data": Output(type="uri_folder")
        },
        environment="<environment name>:1"
    )(raw_data=raw_data)

    #train step
    train_step = command(
        name="train",
        code=".",
        command=(
            "python src/train.py "
            "--data_path ${{inputs.processed_data}} "
            "--model_output_path ${{outputs.model_output}}"
        ),
        inputs={
            "processed_data":  Input(type="uri_folder")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        },
        environment="<environment name>:1"
    )(processed_data=preprocess_step.outputs.processed_data)

    #evaluate step
    evaluate_step = command(
        name="evaluate",
        code=".",
        command=(
            "python src/evaluate.py "
            "--data_path ${{inputs.processed_data}} "
            "--model_dir ${{inputs.model_output}} "
            "--output_path ${{outputs.evaluation_output}}"
        ),
        inputs={
            "processed_data": Input(type="uri_folder"),
            "model_output": Input(type="uri_folder"),
        },
        outputs={
            "evaluation_output": Output(type="uri_folder")
        },
        environment="<environment name>:1"
    )(
        processed_data=preprocess_step.outputs.processed_data,
        model_output=train_step.outputs.model_output
    )

    return {
        "model": train_step.outputs.model_output,
        "metrics": evaluate_step.outputs.evaluation_output
    }


#submit pipeline

pipeline_job = fraud_pipeline(
    raw_data=Input(
        type="uri_file",
        path="<path to raw dataset in blob storage>"
    )
)

submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline submitted: {submitted_job.name}")
