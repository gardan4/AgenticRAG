# src/utils/experiment_tracking.py

import wandb

def init_experiment(project, config, run_name):
    wandb.init(project=project, name=run_name)
    wandb.config.update(config)

def log_metrics(metrics, step=None):
    wandb.log(metrics, step=step)
