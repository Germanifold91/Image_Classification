# Image Classification

# 1. Overview 
The following repository is aimed at demonstrating the integration of [MLflow](https://mlflow.org/docs/latest/index.html) into [Kedro](https://kedro.readthedocs.io) through the use of its plugin, [kedro-mlflow](https://kedro-mlflow.readthedocs.io/en/stable/index.html). This project is NOT intended to serve as a reference for constructing efficient or accurate neural networks. There are numerous resources available that are better suited for that purpose than this repository. However, if you are familiar with either of these two technologies and are looking to learn about the other in the context of what you already know, then this repository will provide a straightforward example to broaden your knowledge.

On the other hand, what this repository WILL DO (or at least, this is the intention) includes:
- An organized and concise methodology to construct machine learning models, from training to inference without leaving aside the evaluation of it.
- How to set up kedro´s catalog so that Mlflow can keep track of the different metric and/or artifacts generated during each training iteration.
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
- `Image_Classification/image-classification/src/image_classification/pipelines/model_training`: As expected the ML pipeline will be the main focus of this file. The metrics generated during the training phase as well as every image/artifact aimed at evaluating the model will be part of this phase, the respective functions for these processes are located at the `training.py` and `predictions.py` scripts.
Those who are familar with kedro will notice that the implementation of such nodes through the pipeline is no different than that on a regular kedro project.

# 5. Catalog 



