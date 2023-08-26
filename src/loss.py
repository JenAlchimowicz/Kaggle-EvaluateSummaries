import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        col_rmse = torch.sqrt(torch.mean((preds - labels) ** 2, dim=0))
        mcrmse = torch.mean(col_rmse)
        return {
            "content_rmse": col_rmse[0].item(),
            "wording_rmse": col_rmse[1].item(),
            "mcrmse": mcrmse,
        }
