from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["images_metadata@pd", "params:split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_image_metadata",
            ),
        ]
    )