import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
        masked_hidden_state = last_hidden_state * expanded_attention_mask
        sum_hidden_state = masked_hidden_state.sum(dim=1)

        token_count = expanded_attention_mask.sum(1)
        token_count = torch.clamp(token_count, min=1e-9)

        mean_embeddings = sum_hidden_state / token_count
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, hf_model_config_path: str = None):
        super().__init__()
        self.cfg = cfg

        if hf_model_config_path:
            self.hf_model_config = torch.load(hf_model_config_path)
            self.encoder = AutoModel.from_config(self.hf_model_config)
        else:
            self.hf_model_config = AutoConfig.from_pretrained(self.cfg.model_name)
            self.encoder = AutoModel.from_pretrained(self.cfg.model_name, config=self.hf_model_config)

        self.pool = MeanPooling()
        if self.cfg.large_sequential:
            self.fc = nn.Sequential(
              nn.Linear(self.hf_model_config.hidden_size, self.cfg.fc_dim),
              nn.ReLU(),
              nn.Linear(self.cfg.fc_dim, 2),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hf_model_config.hidden_size, 2),
            )

    def forward(self, inputs):
        encoder_output = self.encoder(**inputs)
        last_hidden_state = encoder_output["last_hidden_state"]
        pool_feature = self.pool(last_hidden_state, inputs["attention_mask"])
        output = self.fc(pool_feature)
        return output

    def freeze_layers(self):
        named_params = list(self.encoder.named_parameters())
        named_params = named_params[:self.cfg.freeze_layers]
        for name, param in named_params:
            param.requires_grad = False

    def unfreeze_encoder_update_optimizer(self, optimizer):
        if self.cfg.freeze_layers > 0:
            for param in self.encoder.parameters():
                param.requires_grad = True

            optimizer.add_param_group({"params": self.encoder.parameters()})
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.cfg.entire_model_lr
