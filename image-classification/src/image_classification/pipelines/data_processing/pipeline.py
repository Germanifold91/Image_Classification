from kedro.pipeline import Pipeline, node, pipeline

from .nodes import image_registry


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=image_registry,
                inputs="params:path_directory_images",
                outputs="images_metadata@pd",
                name="image_metadata_extraction",
            ),
        ]
    )