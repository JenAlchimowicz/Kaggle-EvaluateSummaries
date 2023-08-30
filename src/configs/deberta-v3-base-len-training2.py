import json
from typing import List

import torch


class Config:
    experiment_name = "deberta-train-len"
    model_name = "microsoft/deberta-v3-base"
    params_checked = "len_training"

    target_cols: List[str] = ["content", "wording"]
    add_prompt_title: bool = True
    add_prompt_question: bool = True
    add_prompt_text: bool = False

    epochs: int = 2
    train_batch_size: int = 4
    val_batch_size: int = 4
    eval_every_n_batches: int = 180
    fc_lr: float = 0.0002
    entire_model_lr: float = 0.000001
    train_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    large_sequential: bool = False
    freeze_layers = 3

    log_file_path = "log.txt"
    neptune_project_name = "jenkaggle/kaggle"
    neptune_api_token = ""

    @classmethod
    def to_dict(cls):
        exclude = set(["to_dict", "to_json", "neptune_api_token"])
        attrs = {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and key not in exclude}
        return attrs

    @classmethod
    def to_json(cls):
        return json.dumps(cls.to_dict())
