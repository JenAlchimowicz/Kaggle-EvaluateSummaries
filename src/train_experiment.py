import gc
import argparse
import importlib

import torch
from accelerate import Accelerator
from dataloaders import TextDataset, load_data
from logger import Logger
from loss import MCRMSELoss
from models import CustomModel
from torch.utils.data import DataLoader
from train_functions import train_epoch, val_epoch, val_epoch_error_analysis
from transformers import AutoTokenizer, DataCollatorWithPadding
import credentials
from optimizer import get_optimizer


def main(cfg):
    train = load_data()
    logger = Logger(cfg)
    logger.run["cfg"] = cfg.to_dict()

    for fold in range(4):
        accelerator = Accelerator()
        model = CustomModel(cfg)
        # model.freeze_layers()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        optimizer = get_optimizer(cfg, model)
        criterion = MCRMSELoss()

        train_df = train[train["fold"] != fold]
        val_df = train[train["fold"] == fold]
        train_df = TextDataset(Config, train_df, tokenizer, True)
        val_df = TextDataset(Config, val_df, tokenizer, False)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(train_df, shuffle=True, batch_size=cfg.train_batch_size, collate_fn=data_collator)
        val_dataloader = DataLoader(val_df, shuffle=False, batch_size=cfg.val_batch_size, collate_fn=data_collator)

        train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, model, optimizer,
        )

        step = 0
        for epoch in range(cfg.epochs):
            # if epoch >= 1:
            #     model.unfreeze_encoder_update_optimizer(optimizer, cfg)

            step, train_epoch_results = train_epoch(
                train_dataloader, optimizer, model, criterion, accelerator, val_dataloader, logger, step, fold, cfg
            )
            val_epoch_results = val_epoch(val_dataloader, model, criterion)
            logger.update_epoch_metrics(fold, epoch, train_epoch_results, val_epoch_results)

        logger.update_fold_metrics()

        top_k_errors = val_epoch_error_analysis(val_dataloader, model, cfg)
        logger.log_error_analysis(top_k_errors, fold)

        del model, train_dataloader, val_dataloader, optimizer, accelerator
        torch.cuda.empty_cache()
        gc.collect()

    logger.update_run_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfn', '--config_file_name', type=str, default='config')
    config_file_name = parser.parse_args().config_file_name

    Config = importlib.import_module(f"configs.{config_file_name}").Config
    cfg = Config()

    print(f"Visible GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    main(cfg)
