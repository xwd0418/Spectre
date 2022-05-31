from argparse import ArgumentParser
from functools import reduce
import torch, os, json, math, torch.nn as nn, pytorch_lightning as pl, numpy as np, scipy.stats as stats, random
import logging
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import TensorBoardLogger
from raytune_models import DSM, Model

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

def train_mnist_tune(config, epochs=5, gpus=1, dir=None):
    model = Model(**config)
    dm = DSM(batch_size=32)
    checkpoint_callback = cb.ModelCheckpoint(monitor="va/mean_loss", mode="min", save_last=True)

    # map reported callback->pl callback
    tune_report = TuneReportCallback({"va/mean_loss": "va/mean_loss"}, on="validation_end")
    
    name = reduce(lambda a, b: f"{a}_({b[0]}={b[1]})", sorted(list(config.items())), "version")

    tbl = TensorBoardLogger(save_dir=dir, version=name, name="raytune_test")
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        progress_bar_refresh_rate = 0,
        logger=tbl,
        callbacks=[
            checkpoint_callback, tune_report
        ])
    trainer.fit(model, datamodule=dm)

def main():
    config = {
        "hparam1": tune.choice([32, 64, 128]),
        "lr": tune.choice([float(f"1e{i}") for i in range(-6, 3)])
    }

    tensorboard_dir = "/workspace/ray_tb"
    ray_dir = "/workspace/ray"

    reporter = CLIReporter(
        parameter_columns=["hparam1", "lr"], 
        metric_columns=["va/mean_loss", "training_iteration", "time_total_s"])

    train_fn_with_parameters = tune.with_parameters(train_mnist_tune, 
        epochs=15, gpus=1, 
        dir=tensorboard_dir)

    resources_per_trial = {"cpu": 1, "gpu": 1}

    search = HyperOptSearch(metric="va/mean_loss", mode="min")

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="va/mean_loss",
        mode="min",
        config=config,
        num_samples=15,
        search_alg=search,
        progress_reporter=reporter,
        local_dir=ray_dir,
        name="my_experiment_1")

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    main()
 