import torch


def get_optimizer(cfg, model):
    encoder_params = list(model.encoder.named_parameters())
    fc_params = list(model.fc.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'lr': cfg.encoder_lr, 'weight_decay': cfg.encoder_weight_decay},
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'lr': cfg.encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in fc_params],
         'lr': cfg.fc_lr, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optimizer_parameters, lr=cfg.encoder_lr)
