"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.model_training import create_pipeline as create_model_training_pipeline



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Assuming this function dynamically finds or manually defines pipelines
    pipelines = {
        "model_training": create_model_training_pipeline(),
        # Add other pipelines as needed
    }

    # Correctly combine all pipelines for the default pipeline
    default_pipeline = Pipeline([])
    for pipeline in pipelines.values():
        default_pipeline += pipeline

    pipelines["__default__"] = default_pipeline

    return pipelines