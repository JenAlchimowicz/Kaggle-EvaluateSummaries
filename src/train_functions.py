from time import time

import numpy as np
import torch
from configs.config import Config


def train_epoch(train_dataloader, optimizer, model, criterion, accelerator, val_dataloader, logger):
    model.train()
    epoch_mcrmse = []
    epoch_content_rmse = []
    epoch_wording_rmse = []
    start_time = time()
    global step
    global fold

    for _i, batch in enumerate(train_dataloader):
        labels = batch["labels"]
        del batch["labels"]

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, labels)
        accelerator.backward(loss["mcrmse"])
        optimizer.step()

        epoch_mcrmse.append(loss["mcrmse"].item())
        epoch_content_rmse.append(loss["content_rmse"])
        epoch_wording_rmse.append(loss["wording_rmse"])

        step += 1
        if step % Config.eval_every_n_batches == 0:
            losses = val_epoch(val_dataloader, model, criterion)
            logger.add_loss_at_step(fold, step, losses)
        # if i >= 2:
        #     break

    return {
        "mcrmse" : np.mean(epoch_mcrmse),
        "content_rmse" : np.mean(epoch_content_rmse),
        "wording_rmse" : np.mean(epoch_wording_rmse),
        "seconds": time() - start_time,
    }

def val_epoch(val_dataloader, model, criterion):
    model.eval()
    epoch_mcrmse = []
    epoch_content_rmse = []
    epoch_wording_rmse = []
    start_time = time()

    for _i, batch in enumerate(val_dataloader):
        labels = batch["labels"]
        del batch["labels"]

        with torch.no_grad():
            output = model(batch)
            loss = criterion(output, labels)

        epoch_mcrmse.append(loss["mcrmse"].item())
        epoch_content_rmse.append(loss["content_rmse"])
        epoch_wording_rmse.append(loss["wording_rmse"])

        # if i >= 2:
        #     break

    return {
        "mcrmse" : np.mean(epoch_mcrmse),
        "content_rmse" : np.mean(epoch_content_rmse),
        "wording_rmse" : np.mean(epoch_wording_rmse),
        "seconds": time() - start_time,
    }
