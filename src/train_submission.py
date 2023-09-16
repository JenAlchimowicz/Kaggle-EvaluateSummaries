import argparse
import importlib
import gc
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from train_functions import train_epoch
from transformers import AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
from tqdm import tqdm

from dataloaders import TextDataset, load_data
from loss import MCRMSELoss
from models import CustomModel
from saver import save_to_s3
from optimizer import get_optimizer


def main(cfg):
    train = load_data()
    logger = None
    fold = None

    accelerator = Accelerator()
    model = CustomModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    optimizer = get_optimizer(cfg, model)
    criterion = MCRMSELoss()

    train_df = train
    val_df = None
    train_df = TextDataset(Config, train_df, tokenizer, True)
    val_df = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_df, shuffle=True, batch_size=cfg.train_batch_size, collate_fn=data_collator)
    val_dataloader = None

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
    )

    step = 0
    losses = defaultdict(list)
    for epoch in tqdm(range(cfg.epochs)):
        step, train_epoch_results = train_epoch(
            train_dataloader, optimizer, model, criterion, accelerator, val_dataloader, logger, step, fold, cfg
        )

        for k, v in train_epoch_results.items():
            losses[f"epoch{epoch}/train_{k}"].append(v)

        torch.cuda.empty_cache()
        gc.collect()

    final_losses = {k: v[-1] for k, v in losses.items()}

    for k, v in final_losses.items():
        print(f"{k}: {v}")

    save_to_s3(model, tokenizer, Config)


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
