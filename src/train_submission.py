import gc
from collections import defaultdict

import torch
from accelerate import Accelerator
from configs.config import Config
from dataloaders import TextDataset, load_data
from loss import MCRMSELoss
from models import CustomModel
from saver import save_to_s3
from torch.utils.data import DataLoader
from train_functions import train_epoch
from transformers import AutoTokenizer, DataCollatorWithPadding

train = load_data()
cfg = Config()

accelerator = Accelerator()
model = CustomModel(cfg)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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


losses = defaultdict(list)
for epoch in range(cfg.epochs):
    train_epoch_results = train_epoch(train_dataloader, optimizer, model, criterion, accelerator)

    for k, v in train_epoch_results.items():
        losses[f"epoch{epoch}/train_{k}"].append(v)

    torch.cuda.empty_cache()
    gc.collect()

final_losses = {k: v[-1] for k, v in losses.items()}

for k, v in final_losses.items():
    print(f"{k}: {v}")

save_to_s3(model, tokenizer, Config)
