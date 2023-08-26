import json
import os
import random
import string
import zipfile
from pathlib import Path

import boto3
import torch


def save_to_s3(model, tokenizer, cfg):
    # Save to disk
    out_dir = f"outputs/{cfg.experiment_name}/experiment_{random_string(6)}"
    torch.save(model.hf_model_config, out_dir + "hf_model_config.pth")
    torch.save(model.state_dict(), out_dir + "weights.pth")
    tokenizer.save_pretrained(out_dir + "tokenizer/")

    cfg_json = cfg.to_json()
    with Path.open(out_dir + "config.json", "w") as file:
        json.dump(cfg_json, file)

    # Zip
    files = [
        out_dir + "hf_model_config.pth",
        out_dir + "weights.pth",
        out_dir + "config.json",
    ]
    directories = [out_dir + "tokenizer"]

    with zipfile.ZipFile(out_dir + "model_bundle.zip", "w") as zipf:
        for file in files:
            filename = Path(file).name
            zipf.write(file, arcname=filename)

        for directory in directories:
            for file in os.listdir(directory):
                zipf.write(os.path.join(directory, file), arcname=os.path.join(os.path.basename(directory), file))


    # Upload to s3
    file_path = out_dir + "/model_bundle.zip"
    key = f"model_bundle_{cfg.experiment_name}.zip"
    upload_to_s3(file_path, key)


def random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_and_digits) for i in range(length))


def upload_to_s3(file_path, key, bucket: str = "kaggle-evaluate-summaries"):
    s3 = boto3.client(
        "s3",
    )
    s3.upload_file(file_path, bucket, key)
