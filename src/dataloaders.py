import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_data():
    summaries_train = pd.read_csv("data/summaries_train.csv")
    prompts_train = pd.read_csv("data/prompts_train.csv")

    train = summaries_train.merge(prompts_train, how="left", on="prompt_id")

    prompt_id_to_fold = {"814d6b": 0, "ebad26": 1, "3b9047": 2, "39c16e": 3}
    train["fold"] = train["prompt_id"].map(prompt_id_to_fold)

    # summaries_test = pd.read_csv("/content/drive/MyDrive/Kaggle-EvaluateSummaries/summaries_test.csv")
    # prompts_test = pd.read_csv("/content/drive/MyDrive/Kaggle-EvaluateSummaries/prompts_test.csv")
    # sample_submission = pd.read_csv("/content/drive/MyDrive/Kaggle-EvaluateSummaries/sample_submission.csv")

    return train


class TextDataset(Dataset):
    def __init__(self, cfg, df: pd.DataFrame, tokenizer, train: bool):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train = train
        self.text = df["text"].values
        self.prompt_title = df["prompt_title"].values
        self.prompt_text = df["prompt_text"].values
        self.prompt_question = df["prompt_question"].values
        self.labels = df[self.cfg.target_cols].values

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        inputs = self.get_transform(index)
        labels = torch.tensor(self.labels[index], dtype=torch.float)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_transform(self, index):
        text = self.text[index]
        prompt_title = self.prompt_title[index]
        prompt_text = self.prompt_text[index]
        prompt_question = self.prompt_question[index]

        if self.train:
            text = self.augment_text(text)

        tokenizer_input = self.combine_texts(text, prompt_title, prompt_text, prompt_question)

        tokenized = self.tokenizer(
            tokenizer_input,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized

    def augment_text(self, text: str) -> str:
        return text

    def combine_texts(self, text, prompt_title, prompt_text, prompt_question) -> str:
        sep = self.tokenizer.sep_token
        final_text = ""
        if self.cfg.add_prompt_title:
            final_text += f"TITLE: {prompt_title} {sep} \n"
        if self.cfg.add_prompt_question:
            final_text += f"QUESTION: {prompt_question} {sep} \n"
        if self.cfg.add_prompt_text:
            final_text += f"CONTEXT: {prompt_text} {sep} \n"

        final_text += f"ANSWER: {text}"
        return final_text

