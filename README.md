# Image Classification

# 1. Overview 
The following repository is aimed at demonstrating the integration of [MLflow](https://mlflow.org/docs/latest/index.html) into [Kedro](https://kedro.readthedocs.io) through the use of its plugin, [kedro-mlflow](https://kedro-mlflow.readthedocs.io/en/stable/index.html). This project is NOT intended to serve as a reference for constructing efficient or accurate neural networks. There are numerous resources available that are better suited for that purpose than this repository. However, if you are familiar with either of these two technologies and are looking to learn about the other in the context of what you already know, then this repository will provide a straightforward example to broaden your knowledge.

On the other hand, what this repository WILL DO (or at least, this is the intention) includes:
- An organized and concise methodology to construct machine learning models, from training to inference without leaving aside the evaluation of it.
- An example of how to set up kedro´s catalog so that Mlflow can keep track of the different metric and/or artifacts generated during each training iteration.
- Integration of kedro-mlflow `pipeline_ml_factory`(**as well as trying to explain what it is**) into kedro's `pipeline_registry.py`.



# 2. Why Kedro MLflow
In the journey from data to deployment in machine learning projects, two main challenges stand out: managing complex data workflows and ensuring the reproducibility of results. As projects scale from experimental stages to production-ready solutions, the need for a structured, efficient approach becomes critical. **Kedro MLflow** addresses these challenges by marrying Kedro’s streamlined data pipeline architecture with MLflow’s comprehensive experiment tracking and model management.

## Simplifying Data Workflows with Kedro
Kedro structures data pipelines in a way that promotes reproducibility, maintainability, and scalability. Its configuration-driven design lets data scientists focus on insights rather than infrastructure, making pipelines clearer and more manageable.

## Enhancing Model Lifecycle with MLflow
MLflow tracks every detail of machine learning experiments, from parameters and metrics to models themselves. This ensures that every experiment is documented, version-controlled, and reproducible, paving the way for transparent and manageable model lifecycle management.

## Unified Workflow
The integration of Kedro with MLflow brings the best of both worlds:

- **Reproducibility & Transparency**: Easily track and reproduce experiments while keeping data pipelines clear and scalable.
- **Modularity & Deployment**: Seamlessly transition from modular data pipelines to production, with every model packaged with its inference pipeline.
- **Efficiency**: Streamline the entire machine learning workflow, reducing overhead and focusing on delivering impactful models.


# 3. Requisites

## 3.1. Datasets
This project makes use of the [Malaria Cell Images Datset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) found at Kaggle. This dataset contains 27.558 images of cells divided into two sub-folders; Infected and Uninfected. 

## 3.2. Clone Repo
``` bash
git clone https://github.com/Germanifold91/Image_Classification
cd Image_Classification
```

## 3.3. Installation of dependencies
```
pip install -r requirements.txt
```

# 4. Pipelines Structures
The execution of this project is divided into two main pipelines:

### Data Processing
- `Image_Classification/image-classification/src/image_classification/pipelines/data_processing`

### Model Training
- `Image_Classification/image-classification/src/image_classification/pipelines/model_training`:
  - As expected the ML pipeline will be the main focus of this file. The metrics generated during the training phase as well as every image/artifact aimed at evaluating the model will be part of this phase, the respective functions for these processes are located at the `training.py` and `predictions.py` scripts.
Those who are familar with kedro will notice that the implementation of such nodes through the pipeline is no different than that on a regular kedro project.

# 5. Catalog 

The introduction of `kedro-mlflow` into Kedros's framework came with the addition of three new `AbstractDataset` for the purpose of metric tracking:
- `MlflowMetricDataset`: Your good old regular metric to measure model performance with a single value such as precission, recall, MSE,......
- `MlflowMetricHistoryDataset`:  Metrics used to track the evolution of a metric during training (eg: validation accuracy in a Neural Network)
- `MlflowMetricsHistoryDataset`:It is a wrapper around a dictionary with metrics which is returned by node and log metrics in MLflow.

```yaml
# General form
my_model_metric:
    type: kedro_mlflow.io.metrics.MlflowMetricHistoryDataset
    run_id: 123456 # OPTIONAL, you should likely let it empty to log in the current run
    key: my_awesome_name # OPTIONAL: if not provided, the dataset name will be used (here "my_model_metric")
    load_args:
        mode: ... # OPTIONAL: "list" by default, one of {"list", "dict", "history"}
    save_args:
        mode: ... # OPTIONAL: "list" by default, one of {"list", "dict", "history"}
```

```yaml
# Case specific
train_acc:
    type: kedro_mlflow.io.metrics.MlflowMetricHistoryDataset
    key: training_accuracy 

train_loss:
    type: kedro_mlflow.io.metrics.MlflowMetricHistoryDataset
    key: training_loss

val_acc:
    type: kedro_mlflow.io.metrics.MlflowMetricHistoryDataset
    key: validation_accuracy 

val_loss:
    type: kedro_mlflow.io.metrics.MlflowMetricHistoryDataset
    key: validation_loss 
```

In addition to these new types of datasets, `kedro-mlflow` defines artifacts as “any data a user may want to track during code execution”. This includes, but is not limited to:

- data needed for the model (e.g encoders, vectorizer, the machine learning model itself…)
- graphs (e.g. ROC or PR curve, importance variables, margins, confusion matrix…)

Artifacts are a very flexible and convenient way to “bind” any data type to your code execution. Mlflow has a two-step process for such binding:

1. Persist the data locally in the desired file format
2. Upload the data to the artifact store

```yaml
# General form
my_dataset_to_version:
    type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
    dataset:
        type: pandas.CSVDataset  # or any valid kedro DataSet
        filepath: /path/to/a/local/destination/file.csv
        load_args:
            sep: ;
        save_args:
            sep: ;
        # ... any other valid arguments for dataset
```

```yaml
# Case Specific
training_evaluation_metrics:
  type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
  dataset:
    type: matplotlib.MatplotlibWriter
    filepath: data/07_model_output/training_metrics/acc_loss_evolution.png

cm_plot:
  type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
  dataset:
    type: matplotlib.MatplotlibWriter
    filepath: data/07_model_output/cm/confusion_matrix.png

roc_plot:
  type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
  dataset:
    type: matplotlib.MatplotlibWriter
    filepath: data/07_model_output/roc/roc_plot.png
```

# 6. Understanding pipeline_ml_factory in Kedro-MLflow

The `pipeline_ml_factory` is a crucial component of the kedro-mlflow integration, designed to streamline and enhance the machine learning development lifecycle within the Kedro framework.

## What is pipeline_ml_factory?

`pipeline_ml_factory` is a function provided by kedro-mlflow that facilitates the integration of MLflow's tracking and model management capabilities with Kedro's data pipelines. It allows you to define a Kedro pipeline that encompasses both the training and inference stages of your machine learning model, automatically handling MLflow logging for model training parameters, metrics, and artifacts.

## How to Use pipeline_ml_factory

Using `pipeline_ml_factory` involves defining two separate pipelines within your Kedro `pipeline_registry.py`: one for model training and another for inference. Such integration within the `register_pipeline()` function can be done as follows;

```python
def register_pipelines() -> Dict[str, Pipeline]:
    """
    Initializes and registers the project's pipelines for data processing, machine learning training,
    and inference, including a combined default pipeline.

    This function creates separate pipelines for data engineering (data_processing), machine learning
    model training (training), and inference, then combines these into a comprehensive machine learning
    pipeline (training_pipeline_ml) with specified training and inference components. The default pipeline
    aggregates all these individual pipelines for ease of use.

    The `training_pipeline_ml` is further customized with MLflow logging configurations.

    Returns:
    - A dictionary mapping pipeline names to their respective `Pipeline` objects, including:
      - 'data_processing': The data engineering pipeline.
      - 'training': The ML training pipeline enhanced with MLflow logging.
      - 'inference': The inference pipeline.
      - '__default__': A combination of all pipelines for comprehensive execution.
    """

    data_processing = data_engineering_pipeline()
    ml_pipeline = model_training_pipeline()
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference") # <------------------ Inference Pipeline
     
    training_pipeline_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"), # <-------------------------- Model Training Pipeline
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
```

## Motivation for Using pipeline_ml_factory

The motivation behind using pipeline_ml_factory in a Kedro-MLflow project is multifold:

### Streamlined Workflow
Integrating training and inference pipelines into a single, cohesive workflow simplifies the process from model development to deployment. pipeline_ml_factory ensures that your model and its preprocessing steps are consistently applied across both stages.

### Enhanced Experiment Tracking
With kedro-mlflow, every aspect of your machine learning experiment, including parameters, metrics, and the model itself, is automatically logged to MLflow. This provides a comprehensive experiment tracking system that facilitates model comparison, versioning, and reproducibility.

### Simplified Model Deployment
Models are logged with their inference pipelines, making them ready for deployment with minimal additional configuration. This integration significantly reduces the overhead typically associated with preparing a model for production.

# 7. Kedro MLflow in action

## 7.1 Pipeline Execution

```bash
kedro run --pipeline data_processing
kedro run --pipeline training
```

## 7.2 MLflow UI
```
kedro mlflow ui
```
