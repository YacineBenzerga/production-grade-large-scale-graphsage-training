# Large-Scale GNN Training Pipeline: GraphSAGE on Amazon Dataset

This repository contains a containerized, high-performance pipeline for training a multi-hop GraphSAGE node classification model on large-scale graph structures (specifically optimized for the Amazon Product/Co-purchase dataset 1.5 million node, ~265 million edges with an average node degree of 168 and 107 classes). The architecture leverages PyTorch Lightning for training orchestration, Optuna for automated hyperparameter tuning, hydra for configuration management and MLflow for robust experiment tracking.


## Project Objectives

The core objectives of this project are twofold:

### Architectural Robustness and Explainability: 

To design a sophisticated Graph Neural Network (GNN) architecture capable of producing highly explainable predictions (instead of using a black box SOTA model), even when subjected to large-scale, severely imbalanced datasets. This is achieved by using appropriate loss function and incorporating structural mechanisms like skip connections, root weight updates, and label correlation tracking to maintain high classification fidelity and transparency.

### Scalable and Efficient Pipeline Orchestration: 

To construct an end-to-end containerized training pipeline optimized for rapid deployment and scaling.

## Start with Notebook

Start by taking a look at [notebook] (https://github.com/YacineBenzerga/production-grade-large-scale-graphsage-training/blob/main/notebooks/Large%20Scale%20Training%20on%20Unbalanced%20data%20using%20GraphSage.md)

## Important Note

Github had recent rendering issues on jupyter notebooks, if you cannot see the notebook properly, there's a rendered markdown version in the same location


## Prerequisites

- Docker & Docker Compose

- NVIDIA Container Toolkit (for GPU acceleration)


### 1. Build the Environment from Scratch

Force a clean build of the custom GNN image to ensure all tracking dependencies and package layers are pristine:

```bash
docker compose build --no-cache
```

### 2. Execute Training Options

#### Option A: Standard Baseline Run

Run a complete baseline training execution using the fixed hyperparameter settings

docker compose up gnn_training

#### Option B: Automated Hyperparameter Search (Optuna)

Execute a multi-trial optimization study by overriding the worker runtime command. This tells Optuna to search for ideal configurations while saving parameters directly to your local tracking database:

Quickly run a single trial with Optuna's default settings:
```bash
docker compose run --rm gnn_training \
  +optuna=hpo \
  optuna.n_trials=1 \
  trainer.max_epochs=1 \
  data.batch_size=15
```

or run a 10-trial study with Optuna's default settings:

```bash
docker compose run --rm gnn_training +optuna=hpo
```

### 3. Launch the Telemetry Dashboard (MLflow)

Spin up the tracking UI server to analyze convergence trends, micro-F1 scores, or parameter importances in real-time:

```bash
docker compose up mlflow_ui
```

Open your browser and navigate to: http://localhost:5050


## Pipeline Monitoring & Telemetry

The system channels all hardware and model statistics into a local SQLite database (mlflow_production.db). The tracking dashboard provides visual tracking for:

train/loss_epoch vs val/loss_epoch: Track network convergence and check for overfitting.

val/f1: Monitor validation Micro-F1 accuracy performance metrics across multiple epochs.

Optuna Parallel Coordinates: Visualize hyperparameter paths (learning rates, batch sizes, neighborhood sample bounds) to identify optimal parameter configurations.




## Configuration Details

### Docker Volume Mapping

The database is kept outside isolated Docker volume structures and saved directly into a shared named voluems named "mlflow_production" which mounts to `/shared_workspace`. This setup ensures that your benchmark logs will never be accidentally erased by running docker volume prune or clearing global container states.

### GPU Reservations

The training worker service automatically hooks into host hardware nodes using standard Docker resource constraints.

To run on a CPU-only architecture, simply comment out the deploy block inside docker-compose.yaml.


## Experimental Results

We were able to achive an validation F1 macro score of ~0.70 using a model that is capable of learning robust structural embeddings on the large-scale Amazon Product graph dataset. We were able to do so by only training on 10% of the entire dataset and we did so using a stratified sampling to account for rare classes.

Telemetry tracked via MLflow demonstrates excellent model generalization and stable convergence, with both training and validation F1-scores scaling up from their initial baselines to peak efficiently without showing signs of overfitting.

The choice of the gnn model, skip connections and the asymetric loss function are all key to achieving this level of performance on a severly imbalanced dataset.

<img width="1805" height="628" alt="Screenshot from 2026-05-29 01-00-12" src="https://github.com/user-attachments/assets/68933072-40af-4e84-bca1-29ea6a7c80d0" />

