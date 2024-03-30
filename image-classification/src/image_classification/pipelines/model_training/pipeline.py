from kedro.pipeline import Pipeline, node
from .training import split_data, create_architecture, evaluation_plots
from .predictions import predict_single_image

def training_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "images_metadata@pd", 
                    "params:split_params"
                    ],
                outputs=[
                    "X_train", 
                    "X_test", 
                    "y_train", 
                    "y_test"
                    ],
                name="split_image_metadata",
                tags=["training", "split"],
            ),
            node(
                func=create_architecture,
                inputs=[
                    "X_train", 
                    "X_test", 
                    "y_test", 
                    "params:nn_training_params"
                    ],
                outputs=[
                    "tf_model", 
                    "train_acc", 
                    "train_loss",
                    "val_acc",
                    "val_loss",
                    "training_evaluation_metrics"
                    ],
                name="model_training_node",
                tags=["training"],
            ),
            node(
                func=evaluation_plots,
                inputs=[
                    "y_test", 
                    "tf_model", 
                    "params:nn_training_params"
                    ],
                outputs=[
                    "cm_plot",
                    "roc_plot"
                    ],
                name="model_evaluation_plots",
                tags=["training", "evaluation"],
            ),
            node(
                func=predict_single_image,
                inputs=[
                    "tf_model", 
                    "params:prediction_params"
                    ],
                outputs="model_predictions",
                name="predict_image",
                tags=["inference"],
            ),
        ]
    )
