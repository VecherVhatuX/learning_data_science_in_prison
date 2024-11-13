import os
import json
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm

@dataclass
class ModelConfig:
    model_identifier: str = "t5-base"
    chat_template: str = "none"
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_rank: int = 64
    lora_target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quant: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_storage_dtype: str = "uint8"
    bnb_4bit_quant_type: str = "nf4"
    use_flash_attention: bool = False
    use_peft_lora: bool = False
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = False
    use_reentrant: bool = False
    use_unsloth: bool = False
    use_triplet_loss_trainer: bool = False

@dataclass
class TrainingDataConfig:
    dataset_name: str = "timdettmers/openassistant-guanaco"
    append_concat_token: bool = False
    add_special_tokens: bool = False
    splits: str = "train,test"
    tokenized_dataset_path: str = None

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    resume_from_checkpoint: str = None

class CustomDataset(Dataset):
    def __init__(self, data, model_args):
        self.data = data
        self.input_ids = np.array([example["input"] if model_args.chat_template == "none" else f"{example['input']} " for example in data])
        self.labels = np.array([example["output"] if model_args.chat_template == "none" else f"{example['output']} " for example in data])
        self.attention_mask = np.array([1]*len(self.input_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}

    @staticmethod
    def load_data(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    @classmethod
    def prepare_datasets(cls, model_args, data_args):
        train_data = cls.load_data("train.json")
        test_data = cls.load_data("test.json")
        return cls(train_data, model_args), cls(test_data, model_args)

    @classmethod
    def create_data_loaders(cls, model_args, data_args):
        train_data, test_data = cls.prepare_datasets(model_args, data_args)
        train_loader = DataLoader(train_data, batch_size=data_args.per_device_train_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=data_args.per_device_eval_batch_size, shuffle=False)
        return train_loader, test_loader

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class T5Model(BaseModel):
    def __init__(self):
        super(T5Model, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def train_step(model, batch, device):
        input_ids = batch["input_ids"].view(1, -1).to(device)
        labels = batch["labels"].view(1, -1).to(device)
        attention_mask = batch["attention_mask"].view(1, -1).to(device)
        outputs = model(input_ids)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, labels)
        return loss

    @classmethod
    def train(cls, model, device, train_loader, num_epochs):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in tqdm(range(num_epochs)):
            for batch in train_loader:
                loss = cls.train_step(model, batch, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    @classmethod
    def run_pipeline(cls, model_args, data_args, training_args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls()
        model.to(device)
        train_loader, _ = CustomDataset.create_data_loaders(model_args, data_args)
        cls.train(model, device, train_loader, training_args.num_train_epochs)

    @classmethod
    def resume_pipeline(cls, model_args, data_args, training_args, checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls()
        model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        train_loader, _ = CustomDataset.create_data_loaders(model_args, data_args)
        cls.train(model, device, train_loader, training_args.num_train_epochs)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none")
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    T5Model.run_pipeline(model_args, data_args, training_args)