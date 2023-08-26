import json
from typing import List

import torch


class Config:
    experiment_name = "roberta-base-train-len"
    model_name = "roberta-base"
    params_checked = "len_training"

    target_cols: List[str] = ["content", "wording"]
    add_prompt_title: bool = True
    add_prompt_question: bool = True
    add_prompt_text: bool = False

    epochs: int = 2
    train_batch_size: int = 16
    val_batch_size: int = 16
    eval_every_n_batches: int = 45
    lr: float = 0.00001
    train_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    log_file_path = "logs/log.txt"
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
