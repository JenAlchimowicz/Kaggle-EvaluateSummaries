from time import time

import numpy as np
import torch


def train_epoch(train_dataloader, optimizer, model, criterion, accelerator,
                val_dataloader, logger, step, fold, cfg):
    model.train()
    epoch_mcrmse = []
    epoch_content_rmse = []
    epoch_wording_rmse = []
    start_time = time()

    for _i, batch in enumerate(train_dataloader):
        labels = batch["labels"]
        del batch["labels"], batch["student_ids"]

        output = model(batch)
        loss = criterion(output, labels)
        loss["mcrmse"] = loss["mcrmse"] / cfg.gradient_accumulation_steps
        accelerator.backward(loss["mcrmse"])

        if (_i + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_mcrmse.append(loss["mcrmse"].item() * cfg.gradient_accumulation_steps)
        epoch_content_rmse.append(loss["content_rmse"])
        epoch_wording_rmse.append(loss["wording_rmse"])

        step += 1
        if step % cfg.eval_every_n_batches == 0:
            losses = val_epoch(val_dataloader, model, criterion)
            logger.add_loss_at_step(fold, step, losses)
            model.train()
        # if _i >= 2:
        #     break

    return step, {
        "mcrmse": np.mean(epoch_mcrmse),
        "content_rmse": np.mean(epoch_content_rmse),
        "wording_rmse": np.mean(epoch_wording_rmse),
        "seconds": time() - start_time,
    }


@torch.no_grad()
def val_epoch(val_dataloader, model, criterion):
    model.eval()
    epoch_mcrmse = []
    epoch_content_rmse = []
    epoch_wording_rmse = []
    start_time = time()

    for _i, batch in enumerate(val_dataloader):
        labels = batch["labels"]
        del batch["labels"], batch["student_ids"]

        output = model(batch)
        loss = criterion(output, labels)

        epoch_mcrmse.append(loss["mcrmse"].item())
        epoch_content_rmse.append(loss["content_rmse"])
        epoch_wording_rmse.append(loss["wording_rmse"])

        # if _i >= 2:
        #     break

    return {
        "mcrmse": np.mean(epoch_mcrmse),
        "content_rmse": np.mean(epoch_content_rmse),
        "wording_rmse": np.mean(epoch_wording_rmse),
        "seconds": time() - start_time,
    }


@torch.no_grad()
def val_epoch_error_analysis(val_dataloader, model, cfg):
    model.eval()
    all_losses = []

    for batch in val_dataloader:
        labels = batch["labels"]
        student_ids = batch["student_ids"]
        del batch["labels"], batch["student_ids"]

        output = model(batch)
        errors = torch.abs(output - labels)
        per_sample_mcrmse = torch.mean(errors, dim=1)

        for i in range(student_ids.shape[0]):
            student_id = student_ids[i].item()
            mcrmse = per_sample_mcrmse[i].item()
            content_label = labels[i][0].item()
            content_pred = output[i][0].item()
            wording_label = labels[i][1].item()
            wording_pred = output[i][1].item()
            all_losses.append([student_id, mcrmse, content_label, content_pred, wording_label, wording_pred])

    sorted_losses = sorted(all_losses, key=lambda x: x[1], reverse=True)
    return sorted_losses[:cfg.log_n_error_samples]
