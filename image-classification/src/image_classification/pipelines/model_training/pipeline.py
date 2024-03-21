from kedro.pipeline import Pipeline, node
from .nodes import split_data, create_architecture

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["images_metadata@pd", "params:split_params"],
                outputs=[
                    "X_train", 
                    "X_test", 
                    "y_train", 
                    "y_test"
                    ],
                name="split_image_metadata",
            ),
            node(
                func=create_architecture,
                inputs=["X_train", "X_test", "y_test", "params:nn_training_params"],
                outputs=[
                    "tf_model", 
                    "train_acc", 
                    "train_loss",
                    "val_acc",
                    "val_loss",
                    "training_evaluation_metrics"
                    ],
                name="model_training_node",
            ),
        ]
    )
