import os
import sys
import json
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification

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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

def load_data(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def process_data(examples, model_args):
    if model_args.chat_template != "none":
        inputs = [f"{example['input']} " for example in examples]
        labels = [f"{example['output']} " for example in examples]
        return {"input_ids": torch.tensor(inputs), "labels": torch.tensor(labels), "attention_mask": torch.tensor([1]*len(inputs))}
    else:
        return {"input_ids": torch.tensor([example["input"] for example in examples]), "labels": torch.tensor([example["output"] for example in examples]), "attention_mask": torch.tensor([1]*len(examples))}

def prepare_datasets(model_args, data_args):
    train_data = load_data("train.json")
    test_data = load_data("test.json")
    train_data = process_data(train_data, model_args)
    test_data = process_data(test_data, model_args)
    return CustomDataset(train_data), CustomDataset(test_data)

def create_data_loaders(model_args, data_args):
    train_data, test_data = prepare_datasets(model_args, data_args)
    train_loader = DataLoader(train_data, batch_size=data_args.per_device_train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=data_args.per_device_eval_batch_size, shuffle=False)
    return train_loader, test_loader

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_args.model_identifier)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1000)

    def forward(self, x):
        outputs = self.model(x["input_ids"], attention_mask=x["attention_mask"])
        x = torch.mean(outputs.last_hidden_state, dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output_layer(x)

class Trainer:
    def __init__(self, model, optimizer, train_loader, epochs, save_path):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.epochs = epochs
        self.save_path = save_path

    def train_step(self, batch):
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = nn.CrossEntropyLoss()(outputs, batch["labels"])
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_loader:
                loss = self.train_step(batch)
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader)}")
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_epoch_{epoch+1}.pth"))

def run_pipeline(model_args, data_args, training_args):
    model = BaseModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, _ = create_data_loaders(model_args, data_args)
    trainer = Trainer(model, optimizer, train_loader, training_args.num_train_epochs, training_args.output_dir)
    trainer.train()

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    model = BaseModel()
    model.load_state_dict(torch.load(checkpoint_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, _ = create_data_loaders(model_args, data_args)
    trainer = Trainer(model, optimizer, train_loader, training_args.num_train_epochs, training_args.output_dir)
    trainer.train()

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none")
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run_pipeline(model_args, data_args, training_args)