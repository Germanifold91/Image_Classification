"""Project pipelines."""
from platform import python_version
from typing import Dict

from kedro_mlflow.pipeline import pipeline_ml_factory
from kedro.pipeline import Pipeline
from image_classification import __version__ as PROJECT_VERSION


from kedro.framework.project import find_pipelines
from .pipelines.data_processing import data_processing as data_engineering_pipeline
from .pipelines.model_training import training_pipeline as model_training_pipeline



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_processing = data_engineering_pipeline()
    ml_pipeline = model_training_pipeline()
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
     
    training_pipeline_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="params:prediction_params",
        log_model_kwargs=dict(
            artifact_path="image_classification",
            conda_env={
                "python": python_version(),
                "build_dependencies": ["pip"],
                "dependencies": [f"image_classification=={PROJECT_VERSION}"],
            },
            signature="auto",
        ),
    )


    return {
        "data_processing": data_processing,
        "training": training_pipeline_ml,
        "inference": inference_pipeline,
        "__default__": data_processing
        + training_pipeline_ml + inference_pipeline
    }
