"""Data Processing pipeline"""

from kedro.pipeline import Pipeline, node, pipeline
from .processing import image_registry


def data_processing(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=image_registry,
                inputs="params:path_directory_images",
                outputs="images_metadata@pd",
                name="image_metadata_extraction",
                tags = ["data_engineering"]
            ),
        ]
    )