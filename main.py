import os
import json
import dataclasses
import typing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

@dataclasses.dataclass
class Config:
    model_id: str = "t5-base"
    chat_format: str = "none"
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_rank: int = 64
    target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization: bool = False
    bit4_compute_type: str = "float16"
    bit4_quant_storage_type: str = "uint8"
    bit4_quant_type: str = "nf4"
    flash_attention: bool = False
    peft_lora: bool = False
    bit8_quantization: bool = False
    bit4_quantization: bool = False
    reentrant: bool = False
    unsloth: bool = False
    triplet_loss_training: bool = False
    dataset_name: str = "timdettmers/openassistant-guanaco"
    append_token: bool = False
    add_special_tokens: bool = False
    data_splits: str = "train,test"
    tokenized_data_path: str = None
    output_path: str = "./results"
    num_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    log_path: str = "./logs"
    save_steps: int = 500
    max_checkpoints: int = 2
    random_seed: int = 42
    resume_checkpoint: str = None

class DatasetImpl:
    def __init__(self, config: Config, data: Dict):
        self.config = config
        self.data = self._prepare(data)

    def _prepare(self, data: Dict) -> Dict:
        return {
            "input_ids": [f"{self.config.chat_format} {example['input']}" for example in data],
            "labels": [f"{self.config.chat_format} {example['output']}" for example in data],
            "attention_mask": [1] * len(data)
        }

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
            "attention_mask": self.data["attention_mask"][idx]
        }

class ModelImpl(nn.Module):
    def __init__(self):
        super(ModelImpl, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        self.fc_out = nn.Linear(128, 1000)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        x = self.fc_out(x)
        return x

class TrainerImpl:
    def __init__(self, config: Config, model: ModelImpl, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    def training_step(self, batch: Dict) -> float:
        inputs, labels = batch["input_ids"], batch["labels"]
        inputs, labels = torch.tensor(inputs, dtype=torch.float32).to(self.device), torch.tensor(labels, dtype=torch.float32).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def training_epoch(self, dataset: DatasetImpl, batch_size: int) -> float:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in data_loader:
            loss = self.training_step(batch)
            total_loss += loss
        return total_loss / len(data_loader)

def _load_dataset(config: Config) -> Tuple[DatasetImpl, DatasetImpl]:
    with open("train.json", 'r') as f:
        train_data = json.load(f)
    with open("test.json", 'r') as f:
        test_data = json.load(f)
    train_dataset = DatasetImpl(config, train_data)
    test_dataset = DatasetImpl(config, test_data)
    return train_dataset, test_dataset

def _main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelImpl()
    model.to(device)
    trainer = TrainerImpl(config, model, device)
    train_dataset, _ = _load_dataset(config)
    for epoch in range(config.num_epochs):
        loss = trainer.training_epoch(train_dataset, config.train_batch_size)
        print(f"Epoch {epoch+1}, Loss: {loss}")

if __name__ == "__main__":
    _main()