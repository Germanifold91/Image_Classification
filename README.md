# Image Classification

# 1. Overview 
The following repository is aimed at showing the integration of [MLflow](https://mlflow.org/docs/latest/index.html) into [Kedro](https://kedro.org) through the use of its plugging [kedro-mlflow](https://kedro-mlflow.readthedocs.io/en/stable/index.html). This project is NOT intented to be used as a reference on how to construct efficient nor accurate neural networks. There are several resources out there destined to do a better job than this repository will ever do in that aspect. If you are familiar with any of these two technologies and you are looking to learn about the other one in the context of what you aleready know then this repository will provide a simple example to expand your knoweldge. 

On the other hand what this repository WILL DO SHOW (or at least this is the intention) is the following:
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
