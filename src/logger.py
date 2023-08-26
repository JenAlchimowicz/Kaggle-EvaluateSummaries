import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import neptune
import numpy as np
from configs.config import Config


def configure_logger(
        logger_name: str,
        log_file_path: Path,
        log_file_level: int = logging.INFO,
        log_stream_level: int = logging.DEBUG,
        format_: str = "%(asctime)s : %(levelname)s : %(name)s : %(message)s",
) -> logging.Logger:
    formatter = logging.Formatter(format_)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_file_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_stream_level)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class Logger:
    def __init__(self, cfg: Config, final_training: bool = False):
        self.cfg = cfg
        self.epoch_metrics = {}
        self.fold_metrics = defaultdict(list)
        self.run_metrics = {}
        self.logger = configure_logger(__name__, cfg.log_file_path)

        self.run = neptune.init_run(
            project=self.cfg.neptune_project_name,
            api_token=self.cfg.neptune_api_token,
        )
        self.run["final_training"] = final_training

    def update_epoch_metrics(self, fold, epoch, train_loss, val_loss):
        for k, v in train_loss.items():
            self.run[f"loss/fold_{fold}/train_{k}"].append(value=v, step=epoch)
            self.epoch_metrics[f"train_{k}"] = v
        for k, v in val_loss.items():
            self.run[f"loss/fold_{fold}/val_{k}"].append(value=v, step=epoch)
            self.epoch_metrics[f"val_{k}"] = v

    def update_fold_metrics(self):
        for k, v in self.epoch_metrics.items():
            self.fold_metrics[k].append(v)
        self.epoch_metrics = {}

    def update_run_metrics(self):
        for k, v in self.fold_metrics.items():
            self.run_metrics[k] = np.mean(v)

        self.run["loss"] = self.run_metrics
        self.run["cfg"] = self.cfg.to_dict()
        self.run.stop()

    def add_loss_at_step(self, fold: int, step: int, losses: Dict[str, float]):
        self.run[f"loss/per_step_loss/fold_{fold}/mcrmse"].append(
            value=losses["mcrmse"], step=step*self.cfg.train_batch_size,
        )
        self.run[f"loss/per_step_loss/fold_{fold}/content_rmse"].append(
            value=losses["content_rmse"], step=step*self.cfg.train_batch_size,
        )
        self.run[f"loss/per_step_loss/fold_{fold}/wording_rmse"].append(
            value=losses["wording_rmse"], step=step*self.cfg.train_batch_size,
        )
